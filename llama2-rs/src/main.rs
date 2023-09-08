use std::mem;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::io::{self, Write};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const CONF_VALS: usize = 7;
const CONF_SIZE: usize = std::mem::size_of::<[i32; CONF_VALS]>();

pub trait LlamaState {
    type Weight;

    fn rms_norm(&mut self, w: &Self::Weight, layer: usize);
    fn self_rms_norm(&mut self, w: &Self::Weight);
    fn add_x_xb(&mut self);
    fn add_x_xb2(&mut self);
}

#[derive(Debug, Clone, Copy)]
struct Config {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
    shared_weights: bool,
}

impl Config {
    /// Read raw bytes and force those to be our config type (which conforms to C mem layout)
    fn from_file(path: &str) -> Self {
        let mut model_bin =
            File::open(path).expect(format!("Couldn't find model file at {}", path).as_str());
        let mut buffer = [0; CONF_SIZE];
        model_bin.read_exact(&mut buffer).unwrap();
        let raw_conf = unsafe { mem::transmute::<[u8; CONF_SIZE], [i32; CONF_VALS]>(buffer) };
        let (vocab_size, shared_weights) = if raw_conf[5] < 0 {
            (-raw_conf[5] as usize, true)
        } else {
            (raw_conf[5] as usize, false)
        };

        Self {
            dim: raw_conf[0] as usize,
            hidden_dim: raw_conf[1] as usize,
            n_layers: raw_conf[2] as usize,
            n_heads: raw_conf[3] as usize,
            n_kv_heads: raw_conf[4] as usize,
            vocab_size,
            seq_len: raw_conf[6] as usize,
            shared_weights,
        }
    }
}

struct Vocab {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}

impl Vocab {
    fn from_file(vocab_size: usize, path: &str) -> Self {
        let mut bytes = Vec::<u8>::new();
        let mut offsets = vec![0usize; 1];
        let mut vocab_bin =
            File::open(path).expect(format!("Couldn't find tokenizer file at {}", path).as_str());
        let mut len = [0; 4];
        let mut val = [0; 1];
        for vs in 0..vocab_size {
            vocab_bin.read_exact(&mut len).unwrap();
            let l = unsafe { mem::transmute::<[u8; 4], i32>(len) };
            offsets.push(offsets.last().unwrap() + l as usize);
            (0..l).for_each(|_| {
                vocab_bin.read_exact(&mut val).unwrap();
                bytes.extend(val);
            });
        }

        assert_eq!(offsets.len(), vocab_size + 1);

        Self { bytes, offsets }
    }

    fn get_token(&self, idx: usize) -> &str {
        let (st, en) = (self.offsets[idx], self.offsets[idx + 1]);
        let b = &self.bytes[st..en];
        std::str::from_utf8(b).unwrap()
    }
}

// generic placeholder
type Ty = f32;

struct Q8Vec {
    factor: Ty,
    vals: Vec<i8>,
}

impl Q8Vec {
    fn dot(&self, other: &Self) -> Ty {
        let common_fact = self.factor * other.factor;
        self.vals
            .iter()
            .zip(other.vals.iter())
            .map(|(x, y)| (x * y) as Ty)
            .fold(0 as Ty, |acc, v| v * common_fact)
    }

    fn quantize(vec: &[Ty]) -> Self {
        let rmax = vec.iter().fold(Ty::NAN, |acc, &v| acc.max(v.abs()));
        let inv_factor = i8::MAX as Ty / rmax;
        let vals = vec.iter().map(|v| (v * inv_factor).round() as i8);
        Self {
            factor: 1 as Ty / inv_factor,
            vals: vals.collect(),
        }
    }

    /// Quantize row wise, takes array of [L, d1, d2] Ty return [L, d1] Q_8Row
    fn quantize_3d(data: &[Ty], layers: usize, rows: usize, cols: usize) -> Vec<Self> {
        let mut out = Vec::with_capacity(layers * rows);
        for layer_mat in data.chunks_exact(rows * cols) {
            for row in layer_mat.chunks_exact(cols) {
                let rmax = row.iter().fold(Ty::NAN, |acc, &v| acc.max(v.abs()));
                let inv_factor = i8::MAX as Ty / rmax;
                let vals = row.iter().map(|v| (v * inv_factor).round() as i8);
                out.push(Self {
                    factor: 1 as Ty / inv_factor,
                    vals: vals.collect(),
                })
            }
        }
        out
    }
}

struct TransformerWeights<Q> {
    /// (vocab_size, dim)
    token_embedding_table: Vec<Ty>,
    /// (layer, dim) rmsnorm weights
    rms_att_weight: Vec<Ty>,
    /// (layer, dim)
    rms_ffn_weight: Vec<Ty>,
    // weights for matmuls
    /// (layer, dim, dim)
    wq: Vec<Q>,
    /// (layer, dim, dim)
    wk: Vec<Q>,
    /// (layer, dim, dim)
    wv: Vec<Q>,
    /// (layer, dim, dim)
    wo: Vec<Q>,
    // weights for ffn
    /// (layer, hidden_dim, dim)
    w1: Vec<Q>,
    /// (layer, dim, hidden_dim)
    w2: Vec<Q>,
    /// (layer, hidden_dim, dim)
    w3: Vec<Q>,
    // final rmsnorm
    /// (dim,)
    rms_final_weight: Vec<Ty>,
    // freq_cis for RoPE relatively positional embeddings
    /// (seq_len, dim/2)
    freq_cis_real: Vec<Ty>,
    /// (seq_len, dim/2)
    freq_cis_imag: Vec<Ty>,
    /// Last layer classifier: (vocab_size, dim)
    wcls: Option<Vec<Ty>>,
}

fn _alloc_and_read(file: &mut File, size: usize) -> Vec<Ty> {
    let bytes_to_read = size * std::mem::size_of::<Ty>();
    let mut raw_w_data = vec![0; bytes_to_read];
    file.read_exact(&mut raw_w_data)
        .expect("Failed to read weights file");
    unsafe {
        let float_ptr = raw_w_data.as_ptr() as *const Ty;
        let data = std::slice::from_raw_parts(float_ptr, size);
        data.to_vec()
    }
}

impl<Q> TransformerWeights<Q> {
    fn _alloc_vecs(cfg: &Config, path: &str) -> ([Vec<Ty>; 13], Option<Vec<Ty>>) {
        let mut model_bin = File::open(path).unwrap();

        model_bin.seek(SeekFrom::Start(CONF_SIZE as u64)).unwrap();

        let mut f = |s: usize| _alloc_and_read(&mut model_bin, s);
        let head_size = cfg.dim / cfg.n_heads;
        (
            [
                f(cfg.vocab_size * cfg.dim),
                f(cfg.n_layers * cfg.dim),
                f(cfg.n_layers * cfg.dim * cfg.dim),
                f(cfg.n_layers * cfg.dim * cfg.dim),
                f(cfg.n_layers * cfg.dim * cfg.dim),
                f(cfg.n_layers * cfg.dim * cfg.dim),
                f(cfg.n_layers * cfg.dim),
                f(cfg.n_layers * cfg.dim * cfg.hidden_dim),
                f(cfg.n_layers * cfg.dim * cfg.hidden_dim),
                f(cfg.n_layers * cfg.dim * cfg.hidden_dim),
                f(cfg.dim),
                f(cfg.seq_len * (head_size / 2)),
                f(cfg.seq_len * (head_size / 2)),
            ],
            cfg.shared_weights.then(|| f(cfg.vocab_size * cfg.dim)),
        )
    }
}

impl TransformerWeights<Ty> {
    fn read_from_file(cfg: &Config, path: &str) -> Self {
        let (vecs, wcls) = Self::_alloc_vecs(cfg, path);

        Self {
            token_embedding_table: vecs[0].to_owned(),
            rms_att_weight: vecs[1].to_owned(),
            wq: vecs[2].to_owned(),
            wk: vecs[3].to_owned(),
            wv: vecs[4].to_owned(),
            wo: vecs[5].to_owned(),
            rms_ffn_weight: vecs[6].to_owned(),
            w1: vecs[7].to_owned(),
            w2: vecs[8].to_owned(),
            w3: vecs[9].to_owned(),
            rms_final_weight: vecs[10].to_owned(),
            freq_cis_real: vecs[11].to_owned(),
            freq_cis_imag: vecs[12].to_owned(),
            wcls,
        }
    }
}

impl TransformerWeights<Q8Vec> {
    fn read_from_file(cfg: &Config, path: &str) -> Self {
        let (vecs, wcls) = Self::_alloc_vecs(cfg, path);
        Self {
            token_embedding_table: vecs[0].to_owned(),
            rms_att_weight: vecs[1].to_owned(),
            wq: Q8Vec::quantize_3d(&vecs[2], cfg.n_layers, cfg.dim, cfg.dim),
            wk: Q8Vec::quantize_3d(&vecs[3], cfg.n_layers, cfg.dim, cfg.dim),
            wv: Q8Vec::quantize_3d(&vecs[4], cfg.n_layers, cfg.dim, cfg.dim),
            wo: Q8Vec::quantize_3d(&vecs[5], cfg.n_layers, cfg.dim, cfg.dim),
            rms_ffn_weight: vecs[6].to_owned(),
            w1: Q8Vec::quantize_3d(&vecs[7], cfg.n_layers, cfg.hidden_dim, cfg.dim),
            w2: Q8Vec::quantize_3d(&vecs[8], cfg.n_layers, cfg.dim, cfg.hidden_dim),
            w3: Q8Vec::quantize_3d(&vecs[9], cfg.n_layers, cfg.hidden_dim, cfg.dim),
            rms_final_weight: vecs[10].to_owned(),
            freq_cis_real: vecs[11].to_owned(),
            freq_cis_imag: vecs[12].to_owned(),
            wcls,
        }
    }
}

struct RunState {
    /// activation at current time stamp (dim,)
    x: Vec<Ty>,
    /// same, but inside a residual branch (dim,)
    xb: Vec<Ty>,
    /// an additional buffer just for convenience (dim,)
    xb2: Vec<Ty>,
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    hb: Vec<Ty>,
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<Ty>,
    /// query (dim,)
    q: Vec<Ty>,
    /// key (dim,)
    k: Vec<Ty>,
    /// value (dim,)
    v: Vec<Ty>,
    /// buffer for scores/attention values (seq_len,)
    att: Vec<Ty>,
    /// output logits
    logits: Vec<Ty>,
    // kv cache
    /// (layer, seq_len, dim)
    key_cache: Vec<Ty>,
    /// (layer, seq_len, dim)
    value_cache: Vec<Ty>,
}

impl LlamaState for RunState {
    type Weight = Vec<Ty>;

    fn rms_norm(&mut self, w: &Self::Weight, layer: usize) {
        let dim = self.x.len();
        let w_it = w
            .chunks_exact(dim)
            .skip(layer)
            .next()
            .expect("Weight should be exact multiple of layers and dim");
        let ss = self.x.iter().fold(0f32, |init, &v| init + v * v) / (self.x.len() as Ty);
        let ss = (1 as Ty) / (ss + 1e-5).sqrt();
        let normed = self.x.iter().zip(w_it).map(|(xx, ww)| xx * ww * ss);
        self.xb
            .iter_mut()
            .zip(normed)
            .for_each(|(dst, src)| *dst = src);
    }

    fn self_rms_norm(&mut self, w: &Self::Weight) {
        let ss = self.x.iter().fold(0f32, |init, &v| init + v * v) / (self.x.len() as Ty);
        let ss = (1 as Ty) / (ss + 1e-5).sqrt();
        self.x
            .iter_mut()
            .zip(w.iter())
            .for_each(|(xx, ww)| (*xx) *= ww * ss);
    }
    #[inline]
    fn add_x_xb2(&mut self) {
        self.x
            .iter_mut()
            .zip(self.xb2.iter())
            .for_each(|(dst, src)| *dst += *src);
    }

    #[inline]
    fn add_x_xb(&mut self) {
        self.x
            .iter_mut()
            .zip(self.xb.iter())
            .for_each(|(dst, src)| *dst += *src);
    }
}

fn rmsnorm(out: &mut [Ty], x: &[Ty], w: &[Ty]) {
    let ss = x.iter().fold(0f32, |init, &v| init + v * v) / (x.len() as Ty);
    let ss = (1 as Ty) / (ss + 1e-5).sqrt();
    let normed = x.iter().zip(w.iter()).map(|(xx, ww)| xx * ww * ss);
    out.iter_mut().zip(normed).for_each(|(dst, src)| *dst = src);
}

fn matmul_q8_ty(out: &mut [Ty], x: &Q8Vec, w: &[Q8Vec]) {
    out.iter_mut().enumerate().for_each(|(i, out_val)| {
        *out_val = unsafe { w.get_unchecked(i).dot(x) };
    });
}

/// For now this is a matvec
/// Wx: [n, d]x[d,] -> [n,]
#[cfg(feature = "parallel")]
fn matmul(out: &mut [Ty], x: &[Ty], w: &[Ty], in_dim: usize) {
    out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
        *out_val = _uncheked_slice(w, i * in_dim, in_dim)
            .iter()
            .zip(x.iter())
            .fold(0 as Ty, |acc, (&_w, &_x)| acc + _w * _x);
    });
}

#[cfg(not(feature = "parallel"))]
fn matmul(out: &mut [Ty], x: &[Ty], w: &[Ty], in_dim: usize) {
    for (row, out_elem) in w.chunks_exact(in_dim).zip(out.iter_mut()) {
        let val = row
            .iter()
            .zip(x.iter())
            .fold(0 as Ty, |acc, (&_w, &_x)| acc + _w * _x);
        *out_elem = val;
    }
}

fn _uncheked_mut_slice(s: &[Ty], offset: usize, size: usize) -> &mut [Ty] {
    let ptr = s.as_ptr() as *mut Ty;
    unsafe {
        let st = ptr.add(offset);
        std::slice::from_raw_parts_mut(st, size)
    }
}

fn _uncheked_slice<Q>(s: &[Q], offset: usize, size: usize) -> &[Q] {
    let ptr = s.as_ptr();
    unsafe {
        let st = ptr.add(offset);
        std::slice::from_raw_parts(st, size)
    }
}

fn inplace_softmax(x: &mut [Ty]) {
    let max_val = x.iter().fold(Ty::NAN, |acc, &v| v.max(acc));
    let mut denom = 0 as Ty;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        denom += *v;
    }

    x.iter_mut().for_each(|v| *v /= denom);
}

fn cdf_sample(probs: &[Ty]) -> usize {
    let mut small_rng = SmallRng::from_entropy();

    let r = small_rng.gen::<Ty>();
    let mut cdf = 0 as Ty;
    for (idx, p) in probs.iter().enumerate() {
        cdf += *p;
        if r < cdf {
            return idx;
        }
    }
    probs.len() - 1
}

fn _att_head_impl(
    layer: usize,
    h: usize,
    q: &[Ty],
    xb: &mut [Ty],
    layer_cached_keys: &[Ty],
    layer_cached_vals: &[Ty],
    att: &mut [Ty],
    pos: usize,
    head_size: usize,
    cfg: &Config,
) {
    let mut head_k_all_pos = layer_cached_keys
        .chunks_exact(head_size)
        .skip(h)
        .step_by(cfg.n_heads);

    for t in 0..=pos {
        let k = head_k_all_pos.next().unwrap();
        let score = k
            .iter()
            .zip(q.iter())
            .fold(0 as Ty, |acc, (_k, _q)| acc + _k * _q);
        let score = score / (head_size as Ty).sqrt();
        unsafe {
            *att.get_unchecked_mut(t) = score;
        }
    }

    let seq_cached_vals = _uncheked_slice(
        layer_cached_vals,
        layer * cfg.dim * cfg.seq_len,
        cfg.seq_len * cfg.dim,
    )
    .chunks_exact(head_size)
    .skip(h)
    .step_by(cfg.n_heads);
    inplace_softmax(&mut att[..=pos]);
    // cahced vals have head_size values in it. we need to go over all t vals and update xb
    // dst is head_size part of xb, we gonna add t (actually pos values) values into it
    let dst = xb;
    // clear dst
    dst.iter_mut().for_each(|v| *v = 0f32);
    // this is different from Karphaty's impl. first go over all head size values, than skip time stamp
    // Why Karphaty's inner loop does C.dim jumps?
    // this goes over time stamps
    for (vals, attn_w) in seq_cached_vals.zip(att.iter()).take(pos + 1) {
        // aggregate timestamp to xb
        for (val, dst) in vals.iter().zip(dst.iter_mut()) {
            *dst += val * attn_w;
        }
    }
}

impl RunState {
    fn init(cfg: &Config) -> Self {
        let f = |size: usize| vec![0 as Ty; size];
        Self {
            x: f(cfg.dim),
            xb: f(cfg.dim),
            xb2: f(cfg.dim),
            hb: f(cfg.hidden_dim),
            hb2: f(cfg.hidden_dim),
            q: f(cfg.dim),
            k: f(cfg.dim),
            v: f(cfg.dim),
            att: f(cfg.seq_len * cfg.n_heads),
            logits: f(cfg.vocab_size),
            key_cache: f(cfg.n_layers * cfg.seq_len * cfg.dim),
            value_cache: f(cfg.n_layers * cfg.seq_len * cfg.dim),
        }
    }

    fn qkv_for_layer(&mut self, l: usize, w: &TransformerWeights<Ty>, dim: usize) {
        let wq = _uncheked_slice(&w.wq, l * dim * dim, dim * dim);
        let wk = _uncheked_slice(&w.wk, l * dim * dim, dim * dim);
        let wv = _uncheked_slice(&w.wv, l * dim * dim, dim * dim);

        matmul(&mut self.q, &self.xb, wq, dim);
        matmul(&mut self.k, &self.xb, wk, dim);
        matmul(&mut self.v, &self.xb, wv, dim);
    }

    fn cache_kv(&mut self, pos: usize, layer: usize, cfg: &Config) {
        let offset = layer * cfg.dim * cfg.seq_len + pos * cfg.dim;
        let kc = _uncheked_mut_slice(&mut self.key_cache, offset, cfg.dim);
        let vc = _uncheked_mut_slice(&mut self.value_cache, offset, cfg.dim);
        // TODO: make theese unsafe and remove len checks
        kc.copy_from_slice(&self.k);
        vc.copy_from_slice(&self.v);
    }

    fn rope(&mut self, pos: usize, w: &TransformerWeights<Ty>, n_heads: usize, dim: usize) {
        let head_size = dim / n_heads;
        let qk_heads = self
            .q
            .chunks_exact_mut(head_size)
            .zip(self.k.chunks_exact_mut(head_size));

        for (q, k) in qk_heads {
            let mut cis_real = w.freq_cis_real[pos * head_size / 2..]
                .iter()
                .take(head_size / 2);
            let mut cis_imag = w.freq_cis_imag[pos * head_size / 2..]
                .iter()
                .take(head_size / 2);

            for (qq, kk) in q.chunks_exact_mut(2).zip(k.chunks_exact_mut(2)) {
                let (q0, q1) = (qq[0], qq[1]);
                let (k0, k1) = (kk[0], kk[1]);
                let fcr = cis_real.next().unwrap();
                let fci = cis_imag.next().unwrap();
                qq[0] = q0 * fcr - q1 * fci;
                qq[1] = q0 * fci + q1 * fcr;
                kk[0] = k0 * fcr - k1 * fci;
                kk[1] = k0 * fci + k1 * fcr;
            }
        }
    }

    fn attention(&mut self, pos: usize, layer: usize, cfg: &Config) {
        assert!(
            pos < cfg.seq_len,
            "Can't attend outside of initialized seq lenght"
        );

        let head_size = cfg.dim / cfg.n_heads;

        // (seq_len, dim)
        let layer_cached_keys = self
            .key_cache
            .chunks_exact(cfg.seq_len * cfg.dim)
            .skip(layer)
            .take(1)
            .next()
            .unwrap();

        let attn_lambda = |h| {
            let q = _uncheked_slice(&self.q, h * head_size, head_size);

            // Brutishly force xb to be mutable slice
            // only a single head is going to be using this part of xb.
            let xb = _uncheked_mut_slice(&self.xb, h * head_size, head_size);
            let att = _uncheked_mut_slice(&self.att, h * cfg.seq_len, cfg.seq_len);
            // let cached_keys =
            let layer_cached_vals =
                _uncheked_slice(&self.value_cache, 0, cfg.n_layers * cfg.seq_len * cfg.dim);

            _att_head_impl(
                layer,
                h,
                q,
                xb,
                layer_cached_keys,
                layer_cached_vals,
                att,
                pos,
                head_size,
                cfg,
            )
        };

        #[cfg(feature = "parallel")]
        (0..cfg.n_heads).into_par_iter().for_each(attn_lambda);

        #[cfg(not(feature = "parallel"))]
        (0..cfg.n_heads).for_each(attn_lambda);
    }

    fn ffn(&mut self, l: usize, w: &TransformerWeights<Ty>, cfg: &Config) {
        // let rms_ffn_w = _uncheked_slice(&w.rms_ffn_weight, l * cfg.dim, cfg.dim);
        // normalize after adding residual
        self.rms_norm(&w.rms_ffn_weight, l);
        // rmsnorm(&mut self.xb, &self.x, rms_ffn_w);

        let w1 = _uncheked_slice(
            &w.w1,
            cfg.dim * cfg.hidden_dim * l,
            cfg.hidden_dim * cfg.dim,
        );
        let w2 = _uncheked_slice(
            &w.w2,
            cfg.dim * cfg.hidden_dim * l,
            cfg.hidden_dim * cfg.dim,
        );
        let w3 = _uncheked_slice(
            &w.w3,
            cfg.dim * cfg.hidden_dim * l,
            cfg.hidden_dim * cfg.dim,
        );
        matmul(&mut self.hb, &self.xb, w1, cfg.dim);
        matmul(&mut self.hb2, &self.xb, w3, cfg.dim);
        // silu on first hidden
        self.hb
            .iter_mut()
            .for_each(|v| *v = (*v) * (1 as Ty / (1 as Ty + (-*v).exp())));

        // mix branches
        self.hb
            .iter_mut()
            .zip(self.hb2.iter())
            .for_each(|(h1, &h2)| *h1 *= h2);

        matmul(&mut self.xb, &self.hb, w2, cfg.hidden_dim);
    }

    fn step(&mut self, token: usize, pos: usize, w: &TransformerWeights<Ty>, cfg: &Config) {
        // copy content row
        // TODO: mayne direct indexing w/o bound checks is faster? benchmark
        w.token_embedding_table
            .chunks_exact(cfg.dim)
            .skip(token)
            .take(1)
            .for_each(|src| self.x.as_mut_slice().copy_from_slice(src));

        for l in 0..cfg.n_layers {
            // let rms_attn_w = _uncheked_slice(&w.rms_att_weight, l * cfg.dim, cfg.dim);
            self.rms_norm(&w.rms_att_weight, l);
            // rmsnorm(&mut self.xb, &self.x, rms_attn_w);

            self.qkv_for_layer(l, w, cfg.dim);

            self.rope(pos, w, cfg.n_heads, cfg.dim);
            self.cache_kv(pos, l, cfg);
            self.attention(pos, l, cfg);
            // self.aggregate_attention(l, pos, C);

            let wo = _uncheked_slice(&w.wo, l * cfg.dim * cfg.dim, cfg.dim * cfg.dim);
            matmul(&mut self.xb2, &self.xb, wo, cfg.dim);
            // post attention residual
            self.add_x_xb2();
            // self.x
            //     .iter_mut()
            //     .zip(self.xb2.iter())
            //     .for_each(|(dst, src)| *dst += *src);

            self.ffn(l, w, cfg);
            // post ffn residual
            self.add_x_xb();
        }

        // last rmsnorm
        self.self_rms_norm(&w.rms_final_weight);

        if let Some(wcls) = &w.wcls {
            matmul(&mut self.logits, &self.x, wcls, cfg.dim);
        } else {
            matmul(&mut self.logits, &self.x, &w.token_embedding_table, cfg.dim);
        }
    }
}

fn main() {
    use std::env;
    use std::time::Instant;

    let model_path = env::args().nth(1).expect("Must pass weights path");
    let temperature = env::args().nth(2).map_or(0 as Ty, |v| {
        v.parse::<Ty>().expect("temperature must be a float")
    });

    let tokenizer_path = "tokenizer.bin";

    let config = Config::from_file(&model_path);
    let seq_len = env::args().nth(3).map_or(config.seq_len, |v| {
        v.parse::<usize>().expect("Sequence len must be integer")
    });

    #[cfg(feature = "parallel")]
    {
        use num_cpus;
        let cpus = num_cpus::get();
        let active_cpus = (cpus).max(1).min(config.n_heads); // use 75% of available cores
        println!("--> [Running Inference on {} CPUs]\n\n", active_cpus);

        rayon::ThreadPoolBuilder::new()
            .num_threads(active_cpus)
            .build_global()
            .unwrap();
    }

    let vocab = Vocab::from_file(config.vocab_size, tokenizer_path);
    let st = Instant::now();
    let weights = TransformerWeights::<Ty>::read_from_file(&config, &model_path);
    println!(
        "--> [Loaded weights in {} secs]\n\n",
        st.elapsed().as_secs()
    );

    let mut benches = vec![];
    for _ in 0..1 {
        let mut state = RunState::init(&config);
        let mut probs = vec![0 as Ty; config.vocab_size];

        let st = Instant::now();
        let mut pos = 0;
        let mut token = 1;
        while pos < seq_len {
            state.step(token, pos, &weights, &config);
            let next = if temperature == 0 as Ty {
                state
                    .logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(index, _)| index)
                    .unwrap()
            } else {
                state
                    .logits
                    .iter()
                    .zip(probs.iter_mut())
                    .for_each(|(logit, p)| *p = logit / temperature);
                inplace_softmax(&mut probs);
                cdf_sample(&probs)
            };
            print!("{}", vocab.get_token(next));
            io::stdout().flush().unwrap();
            pos += 1;
            token = next;
        }
        let ts = pos as f32 / st.elapsed().as_secs_f32();
        benches.push(ts);
    }
    let ts = benches.iter().fold(0f32, |acc, v| acc + v);
    let ts = ts / (benches.len() as f32);

    println!("\n{:.3} Tokens/Sec", ts);
}
