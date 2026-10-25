#!/bin/sh
# Makes the `qwen` backend's three model files in the model store, without
# Python: it downloads Qwen's checkpoints at their pinned revisions,
# checks them, and converts them with this package's Rust converters.
#   - Qwen/Qwen3-ASR-1.7B -> the decoder and mmproj GGUF pair llama.cpp loads.
#     The decoder is byte-identical to ggml-org's Qwen3-ASR-1.7B-GGUF at
#     36a678687ba7; the mmproj differs only in its computed position table.
#   - Qwen/Qwen3-ForcedAligner-0.6B -> the Q8_0 GGUF qwen3-asr.cpp's aligner loads.
# Each checkpoint is kept beside what is made from it, under $PAM_MODELS, else
# ~/Library/Application Support/pam/models, where pam_speech::store looks.
# A file already made and matching its pinned checksum is left alone; a
# changed converter shows as a checksum that fails. Needs curl and cargo.
set -eu
cd "$(dirname "$0")"
models=${PAM_MODELS:-"$HOME/Library/Application Support/pam/models"}

fetch() { # <repo> <revision> <file> <sha256>
  dir="$models/$1/$2"
  mkdir -p "$dir"
  if [ ! -f "$dir/$3" ]; then
    curl -fL --retry 3 -sS -o "$dir/$3.part" "https://huggingface.co/$1/resolve/$2/$3"
    mv "$dir/$3.part" "$dir/$3"
  fi
  [ -z "$4" ] || echo "$4  $dir/$3" | shasum -a 256 -c - >/dev/null
}

made() { # <sha256 lines>: whether every file is there and matches
  echo "$1" | shasum -a 256 -c - >/dev/null 2>&1
}

cargo build --release -q
bin=$(pwd)/target/release

# Qwen3-ASR 1.7B. Its model card's front matter gives the files' licence and tags.
asr=7278e1e70fe206f11671096ffdd38061171dd6e5
dir="$models/Qwen/Qwen3-ASR-1.7B/$asr"
sums="58e22d0532d4eacaf034cfac17a6fed159f37c41390c710186783be439d1fc57  $dir/Qwen3-ASR-1.7B-Q8_0.gguf
b43f64ac7ef3df4d7c5055e478085035e32ee6db6217b05e4f0d3112b405aab1  $dir/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"
if ! made "$sums"; then
  for f in config.json generation_config.json tokenizer_config.json vocab.json merges.txt \
           model.safetensors.index.json; do
    fetch Qwen/Qwen3-ASR-1.7B $asr $f ""
  done
  fetch Qwen/Qwen3-ASR-1.7B $asr model-00001-of-00002.safetensors a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6
  fetch Qwen/Qwen3-ASR-1.7B $asr model-00002-of-00002.safetensors 6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc
  fetch Qwen/Qwen3-ASR-1.7B $asr README.md 5058416891bc47a2051557765997e8c42f8eb78a0e33c3e775bd17d4b0ba4d50
  "$bin/qwen-asr-gguf" "$dir" "$dir/Qwen3-ASR-1.7B-Q8_0.gguf.part" "$dir/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf.part"
  mv "$dir/Qwen3-ASR-1.7B-Q8_0.gguf.part" "$dir/Qwen3-ASR-1.7B-Q8_0.gguf"
  mv "$dir/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf.part" "$dir/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"
  echo "$sums" | shasum -a 256 -c -
fi

# Qwen3-ForcedAligner 0.6B.
aligner=c7cbfc2048c462b0d63a45797104fc9db3ad62b7
dir="$models/Qwen/Qwen3-ForcedAligner-0.6B/$aligner"
out="$dir/qwen3-forced-aligner-0.6b-q8_0.gguf"
sum="2e383c048f11731fc65931e2adf2587377186b4695198e7a0542e42b8cab6373  $out"
if ! made "$sum"; then
  for f in config.json merges.txt vocab.json tokenizer_config.json; do
    fetch Qwen/Qwen3-ForcedAligner-0.6B $aligner $f ""
  done
  fetch Qwen/Qwen3-ForcedAligner-0.6B $aligner model.safetensors 47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7
  "$bin/qwen-aligner-gguf" "$dir" "$out.part" q8_0
  mv "$out.part" "$out"
  echo "$sum" | shasum -a 256 -c -
fi
echo "the qwen backend's files are in $models"
