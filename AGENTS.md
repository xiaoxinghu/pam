# pam

`pam` is the umbrella for Pam's speech tools on macOS: one Cargo workspace, every crate named `pam-*`, and each product installing a short command of its own.

- **`pam`**, the in-room meeting assistant: one laptop and one mic listen live; it separates speakers, names them from introductions, keeps notes, and runs an agent when someone says "Pam, …". `pam listen` runs a meeting in a terminal dashboard; the other subcommands are how the agent reads and writes the meeting.
- **`dictate`**, dictation: it records, transcribes and hands back the text, as JSON lines when piped or as a view on a terminal.

## Crates

| Crate | Holds | Installs |
|---|---|---|
| `crates/pam-speech` | The speech primitives, `Transcriber` and `SpeakerTracker`, their backends, and the model store | |
| `crates/pam-audio` | Audio: the mic resampled to 16 kHz mono, WAV replay, the open input and its pacing, recording, input level | |
| `crates/pam-llama` | llama.cpp linked statically: the one ggml every model in the process runs on, and a small in-process model for the local intent check | |
| `crates/pam-build` | Build-only: pinned source checkouts and CMake builds of the native libraries, for the build scripts of `pam-llama` and `pam-speech` | |
| `crates/pam-core` | The meeting assistant's logic: merge, identity, scribe, invocation, tasks, the meeting folder, and the event and command API | |
| `crates/pam` | The meeting assistant's front-end: the dashboard and the agent's subcommands | `pam` |
| `crates/pam-dictate` | Dictation: the engine and its two faces | `dictate` |

`pam-dictate` depends on `pam-speech` and `pam-audio` only; `pam-core` is the meeting assistant's alone.

`tools/qwen-gguf`, outside the workspace, makes the `qwen` backend's model files: `make.sh` converts Qwen's checkpoints into the model store with its Rust converters. It runs once per machine, so its build stays out of the workspace's.

The real speech backends sit behind cargo features (`nemo-speech`, `qwen`, `llama-cpp`). Without them the `fake` backend replays `.fake.jsonl` scripts, which is how the tests run without a mic or models. A change keeps `cargo build --workspace` and `cargo test --workspace` passing both with the features and without them.

## Code rules

- **Stack:** Rust for the core and the commands, Swift for the native app (later), and pi (TypeScript, Bun-compiled) as the meeting's agent. Production code, including how model files are made, runs without Python. Tests and other tooling are Rust or shell.
- **Logic lives in `pam-core`.** Front-ends (the terminal dashboard now, the Swift app later) draw core events and send core commands, and keep no meeting logic of their own.
- **Speech primitives do one job each:** a `Transcriber` turns speech into words, a `SpeakerTracker` says who spoke when. They know nothing about names or meetings; the core owns IDs, names, timing and merging.
- **Times are 16 kHz sample indices (`Ts = u64`).** Convert to seconds or `mm:ss` only for display and readable files.
