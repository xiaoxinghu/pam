//! Converts Qwen's published Qwen3-ASR checkpoint (safetensors, bf16) into the
//! GGUF pair llama.cpp's multimodal library loads: the decoder (arch `qwen3vl`)
//! and the audio encoder with its projector (`mmproj`, projector `qwen3a`),
//! without Python.
//!
//! It writes what llama.cpp b11553's `convert_hf_to_gguf.py` writes for
//! `Qwen3ASRForConditionalGeneration` (classes `Qwen3ASRTextModel` and
//! `Qwen3ASRMmprojModel`): the same tensor names, types and order, metadata
//! keys in the same order, the same tokenizer arrays, and Q8_0 rounded as
//! gguf-py's `quants.Q8_0` does. ggml-org's published pair was made that way.
//!
//!     qwen-asr-gguf <checkpoint dir> <decoder.gguf> <mmproj.gguf> [q8_0|f16]
//!
//! The checkpoint folder holds Qwen's files at one revision, README.md
//! included: its front matter gives `general.license` and `general.tags`.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use serde_json::Value;

const ALIGNMENT: u64 = 32;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: qwen-asr-gguf <checkpoint dir> <decoder.gguf> <mmproj.gguf> [q8_0|f16]");
    }
    let dir = Path::new(&args[1]);
    let q8 = match args.get(4).map(String::as_str) {
        None | Some("q8_0") => true,
        Some("f16") => false,
        Some(other) => bail!("unknown type {other}"),
    };
    let ck = Checkpoint::open(dir)?;
    decoder(&ck, q8)?.write(Path::new(&args[2]))?;
    mmproj(&ck, q8)?.write(Path::new(&args[3]))?;
    Ok(())
}

/// The checkpoint: config, model card and every tensor, in the order the
/// Python converter meets them (shards by name, then each shard's header).
struct Checkpoint {
    dir: std::path::PathBuf,
    config: Value,
    card: BTreeMap<String, String>,
    generation: Value,
    tensors: Vec<(String, Vec<u64>, Vec<f32>)>,
}

impl Checkpoint {
    fn open(dir: &Path) -> Result<Checkpoint> {
        let read = |name: &str| -> Result<Value> {
            let path = dir.join(name);
            Ok(serde_json::from_str(
                &fs::read_to_string(&path).with_context(|| format!("can't read {}", path.display()))?,
            )?)
        };
        let config = read("config.json")?;
        if config["thinker_config"]["model_type"] != "qwen3_asr" {
            bail!("{} is not a Qwen3-ASR checkpoint", dir.display());
        }
        let index = read("model.safetensors.index.json")?;
        let mut parts: Vec<String> = index["weight_map"]
            .as_object()
            .context("no weight_map")?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        parts.sort();
        parts.dedup();
        let mut tensors = Vec::new();
        for part in &parts {
            let st = fs::read(dir.join(part)).with_context(|| format!("can't read {part}"))?;
            let n = u64::from_le_bytes(st[..8].try_into()?) as usize;
            // serde_json keeps keys sorted; safetensors headers are written sorted too.
            let header: BTreeMap<String, Value> = serde_json::from_slice(&st[8..8 + n])?;
            let data = &st[8 + n..];
            for (name, info) in header {
                if name == "__metadata__" {
                    continue;
                }
                if info["dtype"] != "BF16" {
                    bail!("{name}: expected BF16, got {}", info["dtype"]);
                }
                let shape: Vec<u64> =
                    info["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
                let off: Vec<usize> = info["data_offsets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect();
                let values = data[off[0]..off[1]]
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect();
                tensors.push((name, shape, values));
            }
        }
        Ok(Checkpoint {
            dir: dir.to_path_buf(),
            card: front_matter(&fs::read_to_string(dir.join("README.md")).context("no README.md")?),
            generation: read("generation_config.json")?,
            config,
            tensors,
        })
    }

    fn json(&self, name: &str) -> Result<Value> {
        Ok(serde_json::from_str(&fs::read_to_string(self.dir.join(name))?)?)
    }

    /// `general.*` keys the converter takes from the model card and
    /// generation config, after `general.type` and before the size label's
    /// neighbours: sampling temperature, size label, licence, tags.
    fn general(&self, gguf: &mut Gguf, params: u64) {
        if let Some(t) = self.generation["temperature"].as_f64() {
            gguf.kv("general.sampling.temp", Kv::F32(t as f32));
        }
        gguf.kv("general.size_label", Kv::Str(size_label(params)));
        if let Some(l) = self.card.get("license") {
            gguf.kv("general.license", Kv::Str(l.clone()));
        }
        if let Some(t) = self.card.get("pipeline_tag") {
            gguf.kv("general.tags", Kv::StrArr(vec![t.clone()]));
        }
    }
}

/// The few `key: value` lines of a model card's YAML front matter this needs.
/// Hugging Face's card for Qwen3-ASR opens with an empty block, then the real one.
fn front_matter(readme: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut lines = readme.lines();
    if lines.next().map(str::trim) != Some("---") {
        return out;
    }
    for line in lines {
        if line.trim() == "---" {
            if out.is_empty() {
                continue; // an empty block before the real one
            }
            break;
        }
        if let Some((k, v)) = line.split_once(':')
            && !k.starts_with(' ')
            && !v.trim().is_empty()
        {
            out.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    out
}

/// gguf-py's `model_weight_count_rounded_notation` with two digits.
fn size_label(params: u64) -> String {
    let p = params as f64;
    let (scaled, suffix) = if p > 1e12 {
        (p * 1e-12, "T")
    } else if p > 1e9 {
        (p * 1e-9, "B")
    } else if p > 1e6 {
        (p * 1e-6, "M")
    } else {
        (p * 1e-3, "K")
    };
    let digits = format!("{}", scaled.round_ties_even() as u64).trim_start_matches('0').len();
    format!("{scaled:.prec$}{suffix}", prec = 2usize.saturating_sub(digits))
}

fn decoder(ck: &Checkpoint, q8: bool) -> Result<Gguf> {
    let text = &ck.config["thinker_config"]["text_config"];
    let u = |k: &str| -> Result<u32> {
        text[k].as_u64().map(|x| x as u32).with_context(|| format!("text_config: no {k}"))
    };
    let f = |k: &str| -> Result<f32> {
        text[k].as_f64().map(|x| x as f32).with_context(|| format!("text_config: no {k}"))
    };

    let mut tensors = Vec::new();
    for (name, shape, values) in &ck.tensors {
        let Some(rest) = name.strip_prefix("thinker.") else { continue };
        if rest.starts_with("audio_tower.") {
            continue;
        }
        let gname = text_name(rest).with_context(|| format!("no GGUF name for {name}"))?;
        let ty = if shape.len() <= 1 || gname.ends_with("_norm.weight") {
            Ty::F32
        } else if q8 {
            Ty::Q8_0
        } else {
            Ty::F16
        };
        tensors.push(Tensor::new(gname, shape, values, ty));
    }
    // The text converter writes the tensors outside blocks first, then each block, each group by name.
    tensors.sort_by(|a, b| (block(&a.name), &a.name).cmp(&(block(&b.name), &b.name)));
    let params = tensors.iter().map(|t| t.ne.iter().product::<u64>()).sum();

    let arch = "qwen3vl";
    let mut g = Gguf::default();
    g.kv("general.architecture", Kv::Str(arch.into()));
    g.kv("general.type", Kv::Str("model".into()));
    ck.general(&mut g, params);
    g.kv(&format!("{arch}.block_count"), Kv::U32(u("num_hidden_layers")?));
    g.kv(&format!("{arch}.context_length"), Kv::U32(u("max_position_embeddings")?));
    g.kv(&format!("{arch}.embedding_length"), Kv::U32(u("hidden_size")?));
    g.kv(&format!("{arch}.feed_forward_length"), Kv::U32(u("intermediate_size")?));
    g.kv(&format!("{arch}.attention.head_count"), Kv::U32(u("num_attention_heads")?));
    g.kv(&format!("{arch}.attention.head_count_kv"), Kv::U32(u("num_key_value_heads")?));
    let mut sections: Vec<i32> = text["rope_scaling"]["mrope_section"]
        .as_array()
        .context("no mrope_section")?
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    sections.resize(4, 0);
    g.kv(&format!("{arch}.rope.dimension_sections"), Kv::I32Arr(sections));
    g.kv(&format!("{arch}.rope.freq_base"), Kv::F32(f("rope_theta")?));
    g.kv(&format!("{arch}.attention.layer_norm_rms_epsilon"), Kv::F32(f("rms_norm_eps")?));
    g.kv(&format!("{arch}.attention.key_length"), Kv::U32(u("head_dim")?));
    g.kv(&format!("{arch}.attention.value_length"), Kv::U32(u("head_dim")?));
    g.kv(&format!("{arch}.n_deepstack_layers"), Kv::U32(0));
    tokenizer(ck, u("vocab_size")?, &mut g)?;
    g.kv("general.quantization_version", Kv::U32(2));
    g.kv("general.file_type", Kv::U32(if q8 { 7 } else { 1 }));
    g.tensors = tensors;
    Ok(g)
}

/// The tokenizer as the Python converter writes it for Qwen: every id up to
/// the vocab size, unused ids as `[PAD<id>]`, added tokens CONTROL when
/// special or special-looking and USER_DEFINED otherwise, and `<|im_end|>` as
/// both BOS and EOS.
fn tokenizer(ck: &Checkpoint, vocab_size: u32, g: &mut Gguf) -> Result<()> {
    let vocab: BTreeMap<String, u32> = serde_json::from_value(ck.json("vocab.json")?)?;
    let config = ck.json("tokenizer_config.json")?;
    let added = config["added_tokens_decoder"].as_object().context("no added_tokens_decoder")?;
    let n = vocab_size as usize;
    let mut tokens: Vec<String> = (0..n).map(|i| format!("[PAD{i}]")).collect();
    let mut types = vec![5i32; n]; // UNUSED
    for (t, &id) in &vocab {
        tokens[id as usize] = t.clone();
        types[id as usize] = 1; // NORMAL
    }
    let mut im_end = None;
    for (id, info) in added {
        let id: usize = id.parse()?;
        let content = info["content"].as_str().context("added token without content")?;
        tokens[id] = content.to_string();
        // The converter's `does_token_look_special` also makes `<|…|>` tokens CONTROL
        // when the tokenizer config doesn't mark them special (`<|fim_prefix|>`, …).
        let looks_special = content.starts_with("<|") && content.ends_with("|>")
            || content.starts_with("<｜") && content.ends_with("｜>")
            || content.starts_with("<unused") && content.ends_with('>')
            || ["<pad>", "<mask>", "<2mass>", "[@BOS@]"].contains(&content);
        let special = info["special"].as_bool().unwrap_or(false) || looks_special;
        types[id] = if special { 3 } else { 4 }; // CONTROL, USER_DEFINED
        if content == "<|im_end|>" && im_end.is_none() {
            im_end = Some(id as u32);
        }
    }
    let merges: Vec<String> = fs::read_to_string(ck.dir.join("merges.txt"))?
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with("#version"))
        .map(String::from)
        .collect();
    g.kv("tokenizer.ggml.model", Kv::Str("gpt2".into()));
    g.kv("tokenizer.ggml.pre", Kv::Str("qwen2".into()));
    g.kv("tokenizer.ggml.tokens", Kv::StrArr(tokens));
    g.kv("tokenizer.ggml.token_type", Kv::I32Arr(types));
    g.kv("tokenizer.ggml.merges", Kv::StrArr(merges));
    g.kv("tokenizer.ggml.add_bos_token", Kv::Bool(config["add_bos_token"].as_bool().unwrap_or(false)));
    // Qwen3ASRTextModel replaces the checkpoint's template with plain ChatML.
    g.kv(
        "tokenizer.chat_template",
        Kv::Str("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}".into()),
    );
    let im_end = im_end.context("no <|im_end|> token")?;
    g.kv("tokenizer.ggml.bos_token_id", Kv::U32(im_end));
    g.kv("tokenizer.ggml.eos_token_id", Kv::U32(im_end));
    Ok(())
}

/// llama.cpp's name for a decoder tensor, `thinker.` removed.
fn text_name(hf: &str) -> Option<String> {
    match hf {
        "lm_head.weight" => return Some("output.weight".into()),
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        _ => {}
    }
    let table = [
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
    let (layer, suffix) = hf.strip_prefix("model.layers.")?.split_once('.')?;
    layer.parse::<u32>().ok()?;
    let (_, g) = table.iter().find(|(h, _)| *h == suffix)?;
    Some(format!("blk.{layer}.{g}"))
}

/// A tensor's block number, outside blocks first.
fn block(name: &str) -> Option<u32> {
    name.split('.').find_map(|p| p.parse().ok())
}

fn mmproj(ck: &Checkpoint, q8: bool) -> Result<Gguf> {
    let audio = &ck.config["thinker_config"]["audio_config"];
    let text = &ck.config["thinker_config"]["text_config"];
    let u = |k: &str| -> Result<u32> {
        audio[k].as_u64().map(|x| x as u32).with_context(|| format!("audio_config: no {k}"))
    };
    let d_model = u("d_model")?;

    // The sinusoidal position table the encoder computes, which the converter
    // stores: computed in float32 as Qwen25AudioModel.generate_extra_tensors does.
    let (length, half) = (1500usize, d_model as usize / 2);
    let increment = ((10000f64).ln() / (half - 1) as f64) as f32;
    let inv: Vec<f32> = (0..half).map(|j| (-increment * j as f32).exp()).collect();
    let mut pos = Vec::with_capacity(length * 2 * half);
    for t in 0..length {
        let scaled: Vec<f32> = inv.iter().map(|i| t as f32 * i).collect();
        pos.extend(scaled.iter().map(|x| x.sin()));
        pos.extend(scaled.iter().map(|x| x.cos()));
    }
    let mut tensors =
        vec![Tensor::new("a.position_embd.weight".into(), &[length as u64, 2 * half as u64], &pos, Ty::F32)];

    for (name, shape, values) in &ck.tensors {
        let Some(rest) = name.strip_prefix("thinker.audio_tower.") else { continue };
        let gname = audio_name(rest).with_context(|| format!("no GGUF name for {name}"))?;
        let mut shape = shape.clone();
        if rest.starts_with("conv2d") && rest.ends_with(".bias") {
            shape.extend([1, 1]); // [n] -> [n, 1, 1]
        }
        let ty = if rest.starts_with("conv") && rest.ends_with(".weight") {
            Ty::F16
        } else if shape.len() <= 1 || !gname.ends_with(".weight") {
            Ty::F32
        } else if q8 {
            Ty::Q8_0
        } else {
            Ty::F16
        };
        tensors.push(Tensor::new(gname, &shape, values, ty));
    }
    let params = tensors.iter().map(|t| t.ne.iter().product::<u64>()).sum();

    let mut g = Gguf::default();
    g.kv("general.architecture", Kv::Str("clip".into()));
    g.kv("general.type", Kv::Str("mmproj".into()));
    ck.general(&mut g, params);
    g.kv("general.file_type", Kv::U32(if q8 { 7 } else { 1 }));
    g.kv("clip.has_audio_encoder", Kv::Bool(true));
    g.kv(
        "clip.audio.projection_dim",
        Kv::U32(text["hidden_size"].as_u64().context("no hidden_size")? as u32),
    );
    g.kv("clip.audio.embedding_length", Kv::U32(d_model));
    g.kv("clip.audio.feed_forward_length", Kv::U32(u("encoder_ffn_dim")?));
    g.kv("clip.audio.block_count", Kv::U32(u("encoder_layers")?));
    g.kv("clip.audio.attention.head_count", Kv::U32(u("encoder_attention_heads")?));
    g.kv("clip.audio.num_mel_bins", Kv::U32(u("num_mel_bins")?));
    let eps = audio["layer_norm_eps"].as_f64().unwrap_or(1e-5);
    g.kv("clip.audio.attention.layer_norm_epsilon", Kv::F32(eps as f32));
    g.kv("clip.audio.projector_type", Kv::Str("qwen3a".into()));
    g.kv("general.quantization_version", Kv::U32(2));
    g.tensors = tensors;
    Ok(g)
}

/// llama.cpp's name for an audio-tower tensor, `thinker.audio_tower.` removed.
fn audio_name(hf: &str) -> Option<String> {
    let fixed = [
        ("conv_out.weight", "a.conv_out.weight"),
        ("ln_post.weight", "a.post_ln.weight"),
        ("ln_post.bias", "a.post_ln.bias"),
        ("proj1.weight", "mm.a.mlp.1.weight"),
        ("proj1.bias", "mm.a.mlp.1.bias"),
        ("proj2.weight", "mm.a.mlp.2.weight"),
        ("proj2.bias", "mm.a.mlp.2.bias"),
    ];
    if let Some((_, g)) = fixed.iter().find(|(h, _)| *h == hf) {
        return Some(g.to_string());
    }
    if let Some(rest) = hf.strip_prefix("conv2d") {
        let (n, suffix) = rest.split_once('.')?;
        n.parse::<u32>().ok()?;
        return Some(format!("a.conv2d.{n}.{suffix}"));
    }
    let table = [
        ("fc1", "ffn_up"),
        ("fc2", "ffn_down"),
        ("final_layer_norm", "ln2"),
        ("self_attn_layer_norm", "ln1"),
        ("self_attn.q_proj", "attn_q"),
        ("self_attn.k_proj", "attn_k"),
        ("self_attn.v_proj", "attn_v"),
        ("self_attn.out_proj", "attn_out"),
    ];
    let (layer, rest) = hf.strip_prefix("layers.")?.split_once('.')?;
    layer.parse::<u32>().ok()?;
    let (module, suffix) = rest.rsplit_once('.')?;
    let (_, g) = table.iter().find(|(h, _)| *h == module)?;
    Some(format!("a.blk.{layer}.{g}.{suffix}"))
}

struct Tensor {
    name: String,
    /// GGUF order: fastest dimension first, the reverse of PyTorch's.
    ne: Vec<u64>,
    ty: Ty,
    bytes: Vec<u8>,
}

impl Tensor {
    fn new(name: String, shape: &[u64], values: &[f32], ty: Ty) -> Tensor {
        // Q8_0 needs rows that divide into blocks; gguf-py falls back to F16 otherwise.
        let ty =
            if matches!(ty, Ty::Q8_0) && !shape.last().unwrap().is_multiple_of(32) { Ty::F16 } else { ty };
        let bytes = match ty {
            Ty::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Ty::F16 => values.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect(),
            Ty::Q8_0 => q8_0(values),
        };
        Tensor { name, ne: shape.iter().rev().copied().collect(), ty, bytes }
    }
}

/// gguf-py's Q8_0, "bit-exact with ggml-quants.c": blocks of 32 along each
/// row, an F16 scale of max|x| / 127, and 32 values rounded half away from zero.
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
    Bool(bool),
    Str(String),
    StrArr(Vec<String>),
    I32Arr(Vec<i32>),
}

#[derive(Default)]
struct Gguf {
    kv: Vec<(String, Kv)>,
    tensors: Vec<Tensor>,
}

impl Gguf {
    fn kv(&mut self, key: &str, v: Kv) {
        self.kv.push((key.to_string(), v));
    }

    /// GGUF v3: header, metadata, tensor infos, then the data, each tensor
    /// aligned to 32 bytes.
    fn write(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::new(
            fs::File::create(path).with_context(|| format!("can't create {}", path.display()))?,
        );
        w.write_all(b"GGUF")?;
        w.write_all(&3u32.to_le_bytes())?;
        w.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
        w.write_all(&(self.kv.len() as u64).to_le_bytes())?;
        let mut pos = 24u64;
        for (k, v) in &self.kv {
            pos += write_str(&mut w, k)?;
            pos += v.write(&mut w)?;
        }
        let mut offset = 0u64;
        for t in &self.tensors {
            pos += write_str(&mut w, &t.name)?;
            w.write_all(&(t.ne.len() as u32).to_le_bytes())?;
            for d in &t.ne {
                w.write_all(&d.to_le_bytes())?;
            }
            w.write_all(&(t.ty as u32).to_le_bytes())?;
            w.write_all(&offset.to_le_bytes())?;
            pos += 4 + 8 * t.ne.len() as u64 + 4 + 8;
            offset += pad(t.bytes.len() as u64);
        }
        w.write_all(&vec![0u8; (pad(pos) - pos) as usize])?;
        for t in &self.tensors {
            w.write_all(&t.bytes)?;
            w.write_all(&vec![0u8; (pad(t.bytes.len() as u64) - t.bytes.len() as u64) as usize])?;
        }
        w.flush()?;
        eprintln!("{}: {} tensors, {} keys", path.display(), self.tensors.len(), self.kv.len());
        Ok(())
    }
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
            Kv::Bool(v) => {
                w.write_all(&7u32.to_le_bytes())?;
                w.write_all(&[*v as u8])?;
                5
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
