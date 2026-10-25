# Pam

A meeting assistant for macOS. It listens to the room through one mic, transcribes it live, tells speakers apart, names them when they introduce themselves, keeps notes, and answers questions you type to it.

The same repository has [`dictate`](#dictate), a dictation command: it records, transcribes as you speak, and hands back the text.

## dictate

### Install

You need macOS with:

- the Xcode Command Line Tools (`xcode-select --install`), for clang, git and the linker;
- Rust, from [rustup](https://rustup.rs);
- CMake and Ninja (`brew install cmake ninja`).

Install it with cargo:

```sh
cargo install --locked --git https://github.com/xiaoxinghu/pam --tag dictate-v0.1.0 pam-dictate
```

or with [mise](https://mise.jdx.dev), in `~/.config/mise/config.toml`, then `mise install`:

```toml
[tools]
"cargo:https://github.com/xiaoxinghu/pam" = { version = "tag:dictate-v0.1.0", crate = "pam-dictate" }
```

Either one builds from source. The first build also fetches [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) and [llama.cpp](https://github.com/ggml-org/llama.cpp) and builds them in `~/Library/Caches/pam`. Both are linked into `dictate`, so the binary needs nothing beside it, and you can clear that folder afterwards.

Then download the speech model, once:

```sh
dictate models pull
```

That is `nemotron-en`, about 700 MB, which goes into the model store shared with `pam`: `~/Library/Application Support/pam/models`, or `$PAM_MODELS`. `dictate` never downloads a model on its own.

### Use

```sh
dictate            # speak; Enter prints the text, Escape discards it
dictate | pbcopy   # the same, and the text goes to the clipboard
```

When stdin isn't a terminal, or with `--json`, `dictate` speaks JSON lines instead, for a program to drive: events on stdout, `stop` or `cancel` on stdin. The end of stdin counts as `cancel`, so keep it open. `dictate --help` covers the options, the config file (`~/.config/dictate/config.toml`) and the protocol.

## Using `pam`

```sh
pam listen --title "Stand-up"
```

This listens to the default mic and opens the dashboard. Press `/` to ask Pam something, and `q` to end the meeting. Everything is saved in a folder under `~/Pam/meetings`.

Afterwards, or from another terminal while it runs:

```sh
pam status                  # the meeting at a glance
pam transcript --last 20    # the last 20 lines
pam help                    # every command
```

Try it without a mic or models, using a scripted meeting:

```sh
pam listen --input fixtures/intros.fake.jsonl --realtime --scribe off
```

## Setup

You need Rust, CMake and Ninja (`brew install cmake ninja`), and [pi](https://github.com/earendil-works/pi), logged in, for notes and answers.

```sh
cargo build --release --features nemo-speech,qwen,llama-cpp
```

The first build fetches [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) v0.2.0, [llama.cpp](https://github.com/ggml-org/llama.cpp) b11553 and, for the `qwen` backend, [qwen3-asr.cpp](https://github.com/predict-woo/qwen3-asr.cpp)'s aligner, and builds them in `~/Library/Caches/pam` (set `NEMO_SPEECH_SRC`, `LLAMA_CPP_SRC` or `QWEN3_ASR_DIR` to build your own checkout). They are linked into the binary, which needs nothing beside it but the models: copy `target/release/pam` to any folder on your `PATH`. Without `--features nemo-speech` it can only replay scripted meetings. `qwen` is the Qwen3-ASR transcriber, chosen with `asr = "qwen:qwen3-asr-1.7b"` in the config file; its model files are made, not downloaded: `tools/qwen-gguf/make.sh` converts Qwen's checkpoints into the model store, without Python. llama.cpp runs the local intent check, `pam listen --intent local`; without `llama-cpp`, that option says how to build it. Then download the models your config uses, once:

```sh
pam models pull
```

That is the speech models, and the local intent model (2.7 GB) when `intent_model = "local"`. Models go into the model store, `~/Library/Application Support/pam/models` (or `$PAM_MODELS`), which `pam` shares with `dictate`: each is downloaded once, checked against its pinned SHA-256, and kept where macOS doesn't clear caches. `pam listen` never downloads one itself: a missing model stops it before the meeting, saying what to pull. `pam models list` shows what the store holds and what `pam` uses, `pam models rm <name>` removes a model, and `pam models path` prints the store's folder.

`dictate` has `nemo-speech` by default. From a checkout, `cargo install --path crates/pam-dictate` puts it in `~/.cargo/bin`; add `--features qwen` for the Qwen3-ASR transcriber, whose model is made with `tools/qwen-gguf/make.sh` as for `pam`. `--no-default-features` builds it with only the fake backend.

To build and run in one step, use `cargo pam` in place of `pam`. Cargo flags go before the command:

```sh
cargo pam listen --input fixtures/intros.fake.jsonl --realtime --scribe off
cargo pam --release --features nemo-speech,qwen,llama-cpp listen --title "Demo"
```

`cargo live` is short for that last command; anything after it goes to `listen`:

```sh
cargo live --title "Demo"
```

Optional config in `~/.config/pam/config.toml`:

```toml
mic = "Scarlett Solo USB"   # which input to use; default is the system's
pi = "/opt/pi/pi"           # the pi to run; default is pi on PATH
agent_model = "openrouter/anthropic/claude-sonnet-5.5"   # the model that answers questions
```

For meetings where people speak other languages, one per turn, use the multilingual model (`pam models pull nemotron-3.5` first). It transcribes each line in its own language, but English less well than the default:

```toml
asr = "nemo-speech:nemotron-3.5"
language = "auto"
```

## `pam listen` options

| Option | What it does |
|---|---|
| `--input <mic \| file.wav \| script.fake.jsonl>` | What to listen to. Default `mic`. A `.wav` must be 16 kHz. |
| `--realtime` | Replay a file at real speed instead of as fast as possible. |
| `--title <text>` | The meeting's title. |
| `--attendee <name>` | Someone expected; repeat for each. |
| `--name <name>` | The assistant's name. Default Pam. |
| `--scribe <model \| off>` | The model that writes the notes, through pi. Default `openrouter/anthropic/claude-haiku-5.5`. `off` sends nothing to the cloud. |
| `--intent <model \| local \| off>` | The model that checks whether "Pam, …" is a request: one through pi, `local` for a small model on this Mac, or `off`. Default the scribe's. |
| `--agent <model \| off>` | The model that answers requests, through pi. Default `agent_model` in the config file, else detected from your pi login. |

## Dashboard keys

| Key | Action |
|---|---|
| `/` | Ask Pam something; `Enter` sends |
| `t` | Show or hide the transcript |
| `↑` `↓` | Select a line |
| `←` `→` | Select a speaker |
| `r` | Rename the selected speaker |
| `m` | Merge the selected speaker into another, or move the selected line to another speaker |
| `u` | Undo the latest rename, merge or move |
| `Esc` | Clear the selection or cancel |
| `q` | End the meeting |

## Reading a meeting

These read the latest meeting, or the one given with `--meeting <folder>`. Add `--json` for machine-readable output. `pam help <command>` shows each one's options and examples.

| Command | Shows |
|---|---|
| `pam status` | Elapsed time, people, line count, open items |
| `pam context` | The context pack: people, open items, events and the last few minutes |
| `pam people` | Each speaker: ID, name, talk time, lines |
| `pam transcript` | The lines, filtered by `--last`, `--since`, `--from`/`--to`, `--speaker` |
| `pam get <ref>` | Exact lines (`u0412`, `u0400..u0412`), events (`e037`) or speakers (`spk_3`) |
| `pam around <ref>` | The lines around a line or event |
| `pam search <words>` | Lines and events containing the words |
| `pam events` | The scribe's decisions, actions, questions and so on |
| `pam note <text>` | Add a note |
| `pam say <text>` | Show a message on the stage; the agent answers with it |

## The meeting folder

Each meeting gets its own folder under `~/Pam/meetings` (set `PAM_MEETINGS_ROOT` to change the root) with:

- `transcript.md` — the transcript with names
- `notes.md` — the notes
- `context.md` — a short summary for an agent to read first
- `audio.wav` — the recording
- `tasks` — a folder per question, with the agent's work and answer

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
