//! Converts Qwen's published Qwen3-ForcedAligner-0.6B checkpoint (safetensors,
//! bf16) into the single GGUF file qwen3-asr.cpp's `ForcedAligner` loads,
//! without Python.
//!
//! It writes what the runtime's own `scripts/convert_hf_to_gguf.py` writes:
//! the same tensor names, metadata keys and tokenizer arrays; 1-D tensors as
//! F32 and the rest as F16, or as Q8_0 except embeddings, norms, the
//! timestamp head and the decoder's attention output projections, which stay
//! F16 (the Python converter keeps any name containing `output.weight`).
//!
//!     qwen-aligner-gguf <checkpoint dir> <out.gguf> [f16|q8_0]

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use serde_json::Value;

const ARCH: &str = "qwen3-asr";
const ALIGNMENT: u64 = 32;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: qwen-aligner-gguf <checkpoint dir> <out.gguf> [f16|q8_0]");
    }
    let dir = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    let q8 = match args.get(3).map(String::as_str) {
        None | Some("f16") => false,
        Some("q8_0") => true,
        Some(other) => bail!("unknown type {other}"),
    };

    let config: Value = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
    let thinker = &config["thinker_config"];
    if thinker["model_type"] != "qwen3_forced_aligner" {
        bail!("{} is not a Qwen3-ForcedAligner checkpoint", dir.display());
    }
    let audio = &thinker["audio_config"];
    let text = &thinker["text_config"];
    let u = |v: &Value, k: &str| -> Result<u32> {
        v[k].as_u64().map(|x| x as u32).with_context(|| format!("config: no {k}"))
    };
    let f = |v: &Value, k: &str| -> Result<f32> {
        v[k].as_f64().map(|x| x as f32).with_context(|| format!("config: no {k}"))
    };

    let mut kv: Vec<(String, Kv)> = Vec::new();
    let mut put = |k: &str, v: Kv| kv.push((k.to_string(), v));
    put("general.architecture", Kv::Str(ARCH.into()));
    put("general.type", Kv::Str("model".into()));
    put("general.name", Kv::Str("Qwen3-ForcedAligner-0.6B".into()));
    put("general.file_type", Kv::U32(if q8 { 7 } else { 1 }));
    put("general.quantization_version", Kv::U32(2));
    put(&format!("{ARCH}.block_count"), Kv::U32(u(text, "num_hidden_layers")?));
    put(&format!("{ARCH}.embedding_length"), Kv::U32(u(text, "hidden_size")?));
    put(&format!("{ARCH}.feed_forward_length"), Kv::U32(u(text, "intermediate_size")?));
    put(&format!("{ARCH}.attention.head_count"), Kv::U32(u(text, "num_attention_heads")?));
    put(&format!("{ARCH}.attention.head_count_kv"), Kv::U32(u(text, "num_key_value_heads")?));
    put(&format!("{ARCH}.attention.key_length"), Kv::U32(u(text, "head_dim")?));
    put(&format!("{ARCH}.attention.value_length"), Kv::U32(u(text, "head_dim")?));
    put(&format!("{ARCH}.rope.freq_base"), Kv::F32(f(text, "rope_theta")?));
    put(&format!("{ARCH}.attention.layer_norm_rms_epsilon"), Kv::F32(f(text, "rms_norm_eps")?));
    let vocab_size = u(text, "vocab_size")?;
    put(&format!("{ARCH}.vocab_size"), Kv::U32(vocab_size));
    put(&format!("{ARCH}.audio.encoder.layer_count"), Kv::U32(u(audio, "encoder_layers")?));
    put(&format!("{ARCH}.audio.encoder.embedding_length"), Kv::U32(u(audio, "d_model")?));
    put(&format!("{ARCH}.audio.encoder.attention.head_count"), Kv::U32(u(audio, "encoder_attention_heads")?));
    put(&format!("{ARCH}.audio.encoder.feed_forward_length"), Kv::U32(u(audio, "encoder_ffn_dim")?));
    put(&format!("{ARCH}.audio.num_mel_bins"), Kv::U32(u(audio, "num_mel_bins")?));
    put(&format!("{ARCH}.audio.conv_channels"), Kv::U32(u(audio, "downsample_hidden_size")?));
    put(&format!("{ARCH}.audio.start_token_id"), Kv::U32(u(thinker, "audio_start_token_id")?));
    put(&format!("{ARCH}.audio.end_token_id"), Kv::U32(u(thinker, "audio_end_token_id")?));
    put(&format!("{ARCH}.audio.pad_token_id"), Kv::U32(u(thinker, "audio_token_id")?));
    put(&format!("{ARCH}.classify_num"), Kv::U32(u(thinker, "classify_num")?));
    put(&format!("{ARCH}.timestamp_token_id"), Kv::U32(u(&config, "timestamp_token_id")?));
    put(&format!("{ARCH}.timestamp_segment_time"), Kv::U32(80));
    tokenizer(dir, vocab_size, &mut put)?;

    // The tensors, renamed and converted, in name order.
    let st = fs::read(dir.join("model.safetensors"))?;
    let header_len = u64::from_le_bytes(st[..8].try_into()?) as usize;
    let header: BTreeMap<String, Value> = serde_json::from_slice(&st[8..8 + header_len])?;
    let data = &st[8 + header_len..];
    let mut tensors = Vec::new();
    for (name, info) in &header {
        if name == "__metadata__" {
            continue;
        }
        let Some(gname) = map_name(name) else {
            eprintln!("skipping unmapped tensor {name}");
            continue;
        };
        if info["dtype"] != "BF16" {
            bail!("{name}: expected BF16, got {}", info["dtype"]);
        }
        let shape: Vec<u64> = info["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
        let off: Vec<usize> =
            info["data_offsets"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
        let values: Vec<f32> = data[off[0]..off[1]]
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        let (ty, bytes) = if shape.len() <= 1 {
            (Ty::F32, values.iter().flat_map(|v| v.to_le_bytes()).collect())
        } else if q8 && quantizes(&gname) && shape.last().unwrap().is_multiple_of(32) {
            (Ty::Q8_0, q8_0(&values))
        } else {
            (Ty::F16, values.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect())
        };
        // GGUF lists dimensions fastest first: the reverse of PyTorch's.
        let ne: Vec<u64> = shape.iter().rev().copied().collect();
        tensors.push((gname, ne, ty, bytes));
    }
    eprintln!("{} tensors", tensors.len());

    let mut w = BufWriter::new(fs::File::create(out)?);
    w.write_all(b"GGUF")?;
    w.write_all(&3u32.to_le_bytes())?;
    w.write_all(&(tensors.len() as u64).to_le_bytes())?;
    w.write_all(&(kv.len() as u64).to_le_bytes())?;
    let mut pos = 24u64;
    for (k, v) in &kv {
        pos += write_str(&mut w, k)?;
        pos += v.write(&mut w)?;
    }
    let mut offset = 0u64;
    for (name, ne, ty, bytes) in &tensors {
        pos += write_str(&mut w, name)?;
        w.write_all(&(ne.len() as u32).to_le_bytes())?;
        for d in ne {
            w.write_all(&d.to_le_bytes())?;
        }
        w.write_all(&(*ty as u32).to_le_bytes())?;
        w.write_all(&offset.to_le_bytes())?;
        pos += 4 + 8 * ne.len() as u64 + 4 + 8;
        offset += pad(bytes.len() as u64);
    }
    w.write_all(&vec![0u8; (pad(pos) - pos) as usize])?;
    for (_, _, _, bytes) in &tensors {
        w.write_all(bytes)?;
        w.write_all(&vec![0u8; (pad(bytes.len() as u64) - bytes.len() as u64) as usize])?;
    }
    w.flush()?;
    Ok(())
}

/// The tokenizer as the Python converter writes it: every id up to the vocab
/// size, unused ids as `[PAD<id>]`, added tokens from `tokenizer_config.json`.
fn tokenizer(dir: &Path, vocab_size: u32, put: &mut impl FnMut(&str, Kv)) -> Result<()> {
    let vocab: BTreeMap<String, u32> = serde_json::from_str(&fs::read_to_string(dir.join("vocab.json"))?)?;
    let config: Value = serde_json::from_str(&fs::read_to_string(dir.join("tokenizer_config.json"))?)?;
    let added = config["added_tokens_decoder"].as_object().context("no added_tokens_decoder")?;
    let max_id = vocab.values().copied().chain(added.keys().map(|k| k.parse().unwrap())).max().unwrap_or(0);
    let n = (vocab_size).max(max_id + 1) as usize;
    let mut tokens: Vec<String> = (0..n).map(|i| format!("[PAD{i}]")).collect();
    let mut types = vec![5i32; n]; // UNUSED
    let mut ids: BTreeMap<String, u32> = vocab.clone();
    for (t, &id) in &vocab {
        tokens[id as usize] = t.clone();
        types[id as usize] = 1; // NORMAL
    }
    for (id, info) in added {
        let id: usize = id.parse()?;
        let Some(content) = info["content"].as_str() else { continue };
        tokens[id] = content.to_string();
        types[id] = if info["special"].as_bool().unwrap_or(false) { 3 } else { 1 }; // CONTROL
        ids.insert(content.to_string(), id as u32);
    }
    let merges: Vec<String> = fs::read_to_string(dir.join("merges.txt"))?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();
    put("tokenizer.ggml.model", Kv::Str("gpt2".into()));
    put("tokenizer.ggml.pre", Kv::Str("qwen2".into()));
    put("tokenizer.ggml.tokens", Kv::StrArr(tokens));
    put("tokenizer.ggml.token_type", Kv::I32Arr(types));
    put("tokenizer.ggml.merges", Kv::StrArr(merges));
    let special =
        |k: &str| config[k].as_str().or_else(|| config[k]["content"].as_str()).and_then(|t| ids.get(t));
    if let Some(&id) = special("eos_token") {
        put("tokenizer.ggml.eos_token_id", Kv::U32(id));
    }
    if let Some(&id) = special("pad_token") {
        put("tokenizer.ggml.padding_token_id", Kv::U32(id));
    }
    if let Some(t) = config["chat_template"].as_str() {
        put("tokenizer.chat_template", Kv::Str(t.into()));
    }
    Ok(())
}

/// The runtime's tensor name for a checkpoint tensor.
fn map_name(hf: &str) -> Option<String> {
    let fixed = [
        ("thinker.audio_tower.conv2d1.weight", "audio.encoder.conv1.weight"),
        ("thinker.audio_tower.conv2d1.bias", "audio.encoder.conv1.bias"),
        ("thinker.audio_tower.conv2d2.weight", "audio.encoder.conv2.weight"),
        ("thinker.audio_tower.conv2d2.bias", "audio.encoder.conv2.bias"),
        ("thinker.audio_tower.conv2d3.weight", "audio.encoder.conv3.weight"),
        ("thinker.audio_tower.conv2d3.bias", "audio.encoder.conv3.bias"),
        ("thinker.audio_tower.conv_out.weight", "audio.encoder.conv_out.weight"),
        ("thinker.audio_tower.conv_out.bias", "audio.encoder.conv_out.bias"),
        ("thinker.audio_tower.layer_norm.weight", "audio.encoder.ln.weight"),
        ("thinker.audio_tower.layer_norm.bias", "audio.encoder.ln.bias"),
        ("thinker.audio_tower.ln_post.weight", "audio.encoder.ln_post.weight"),
        ("thinker.audio_tower.ln_post.bias", "audio.encoder.ln_post.bias"),
        ("thinker.audio_tower.embed_positions.weight", "audio.encoder.pos_embd.weight"),
        ("thinker.audio_tower.proj1.weight", "audio.encoder.proj1.weight"),
        ("thinker.audio_tower.proj1.bias", "audio.encoder.proj1.bias"),
        ("thinker.audio_tower.proj2.weight", "audio.encoder.proj2.weight"),
        ("thinker.audio_tower.proj2.bias", "audio.encoder.proj2.bias"),
        ("thinker.model.embed_tokens.weight", "token_embd.weight"),
        ("thinker.model.norm.weight", "output_norm.weight"),
        ("thinker.lm_head.weight", "output.weight"),
        ("thinker.classify_head.weight", "classify_head.weight"),
        ("thinker.classify_head.bias", "classify_head.bias"),
    ];
    if let Some((_, g)) = fixed.iter().find(|(h, _)| *h == hf) {
        return Some(g.to_string());
    }
    let audio = [
        ("self_attn.q_proj.weight", "attn_q.weight"),
        ("self_attn.k_proj.weight", "attn_k.weight"),
        ("self_attn.v_proj.weight", "attn_v.weight"),
        ("self_attn.out_proj.weight", "attn_out.weight"),
        ("self_attn.q_proj.bias", "attn_q.bias"),
        ("self_attn.k_proj.bias", "attn_k.bias"),
        ("self_attn.v_proj.bias", "attn_v.bias"),
        ("self_attn.out_proj.bias", "attn_out.bias"),
        ("self_attn_layer_norm.weight", "attn_norm.weight"),
        ("self_attn_layer_norm.bias", "attn_norm.bias"),
        ("final_layer_norm.weight", "ffn_norm.weight"),
        ("final_layer_norm.bias", "ffn_norm.bias"),
        ("fc1.weight", "ffn_up.weight"),
        ("fc1.bias", "ffn_up.bias"),
        ("fc2.weight", "ffn_down.weight"),
        ("fc2.bias", "ffn_down.bias"),
    ];
    let text = [
        ("input_layernorm.weight", "attn_norm.weight"),
        ("self_attn.q_proj.weight", "attn_q.weight"),
        ("self_attn.k_proj.weight", "attn_k.weight"),
        ("self_attn.v_proj.weight", "attn_v.weight"),
        ("self_attn.o_proj.weight", "attn_output.weight"),
        ("self_attn.q_norm.weight", "attn_q_norm.weight"),
        ("self_attn.k_norm.weight", "attn_k_norm.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ("mlp.gate_proj.weight", "ffn_gate.weight"),
        ("mlp.up_proj.weight", "ffn_up.weight"),
        ("mlp.down_proj.weight", "ffn_down.weight"),
    ];
    for (prefix, out, table) in [
        ("thinker.audio_tower.layers.", "audio.encoder.blk.", &audio[..]),
        ("thinker.model.layers.", "blk.", &text[..]),
    ] {
        if let Some(rest) = hf.strip_prefix(prefix) {
            let (layer, suffix) = rest.split_once('.')?;
            layer.parse::<u32>().ok()?;
            let (_, g) = table.iter().find(|(h, _)| *h == suffix)?;
            return Some(format!("{out}{layer}.{g}"));
        }
    }
    None
}

/// Whether Q8_0 applies: not to embeddings, norms, biases or the timestamp head.
fn quantizes(name: &str) -> bool {
    !(["token_embd", "output.weight", "pos_embd", "_norm", ".ln", "ln_post", ".bias"]
        .iter()
        .any(|s| name.contains(s)))
}

/// ggml's Q8_0: blocks of 32 along each row, an F16 scale of max|x| / 127 and
/// 32 rounded `i8`s. A tensor whose rows don't divide into blocks stays F16,
/// as the Python converter's fallback leaves it.
fn q8_0(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() / 32 * 34);
    for block in values.chunks_exact(32) {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend(block.iter().map(|v| (v * id).round() as i8 as u8));
    }
    out
}

#[derive(Clone, Copy)]
enum Ty {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
}

enum Kv {
    U32(u32),
    F32(f32),
    Str(String),
    StrArr(Vec<String>),
    I32Arr(Vec<i32>),
}

impl Kv {
    fn write(&self, w: &mut impl Write) -> Result<u64> {
        Ok(match self {
            Kv::U32(v) => {
                w.write_all(&4u32.to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
                8
            }
            Kv::F32(v) => {
                w.write_all(&6u32.to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
                8
            }
            Kv::Str(s) => {
                w.write_all(&8u32.to_le_bytes())?;
                4 + write_str(w, s)?
            }
            Kv::StrArr(items) => {
                w.write_all(&9u32.to_le_bytes())?;
                w.write_all(&8u32.to_le_bytes())?;
                w.write_all(&(items.len() as u64).to_le_bytes())?;
                let mut n = 16;
                for s in items {
                    n += write_str(w, s)?;
                }
                n
            }
            Kv::I32Arr(items) => {
                w.write_all(&9u32.to_le_bytes())?;
                w.write_all(&5u32.to_le_bytes())?;
                w.write_all(&(items.len() as u64).to_le_bytes())?;
                for v in items {
                    w.write_all(&v.to_le_bytes())?;
                }
                16 + 4 * items.len() as u64
            }
        })
    }
}

fn write_str(w: &mut impl Write, s: &str) -> Result<u64> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(8 + s.len() as u64)
}

fn pad(n: u64) -> u64 {
    n.div_ceil(ALIGNMENT) * ALIGNMENT
}
