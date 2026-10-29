//! Arguments and help for `dictate`.
//!
//! Hand-written, as `pam`'s are, so that every usage error can say what went
//! wrong, show the full usage and give worked examples.

use std::fmt::Write as _;
use std::path::PathBuf;

use pam_speech::{SAMPLE_RATE, Ts};

use crate::config::Config;
use crate::engine::{Input, Options};

/// One option, as the usage block lists it.
struct Arg {
    name: &'static str,
    /// Whether it takes a value (`--timeout 30`) or is a switch.
    value: bool,
    /// The config file's key it overrides or adds to, if any.
    key: &'static str,
    help: &'static str,
}

const fn opt(name: &'static str, key: &'static str, help: &'static str) -> Arg {
    Arg { name, value: true, key, help }
}
const fn switch(name: &'static str, help: &'static str) -> Arg {
    Arg { name, value: false, key: "", help }
}

const USAGE: &str =
    "dictate [--input mic|-|<file.wav>|<script.fake.jsonl>] [--realtime] [--device <name>] [--asr <backend:model>]
               [--language <code>] [--hint <word>]... [--hints <file>] [--timeout <seconds>] [--json]
       dictate models list | pull [<name>...] | rm <name>... | path [--json]";

const ARGS: &[Arg] = &[
    opt(
        "--input",
        "",
        "what to listen to: mic, the default; -, raw 16 kHz mono 32-bit float samples (f32le) on stdin, which then carries no commands; a 16 kHz recording (*.wav); or a fake script (*.fake.jsonl), which is its own transcriber. Anything but the mic runs as fast as the transcriber goes, and its end is a stop",
    ),
    switch(
        "--realtime",
        "take stdin, a recording or a script at real-time speed, as if spoken, rather than as fast as the transcriber goes",
    ),
    opt("--device", "mic", "the input device's name, with --input mic (default: the system's default input)"),
    opt("--asr", "asr", "the transcriber, as <backend>:<model> (default nemo-speech:nemotron-en)"),
    opt("--language", "language", "what is spoken, as a language code (default en-US), or auto"),
    opt(
        "--hint",
        "hints",
        "a name or word the transcriber should favour; repeat for each; adds to the hints key",
    ),
    opt(
        "--hints",
        "hints",
        "a file of hints, one per line; blank lines and lines starting with # are skipped; adds to the hints key",
    ),
    opt(
        "--timeout",
        "timeout",
        "stop on its own after this many seconds, as if `stop` arrived; 0 means never (the default)",
    ),
    switch("--json", "speak JSON lines on stdout even when stdin is a terminal, instead of drawing the view"),
];

const EXAMPLES: &[(&str, &str)] = &[
    (
        "dictate --device \"Yeti Stereo Microphone\" --timeout 60",
        "dictate through a given mic, a minute at most",
    ),
    ("dictate | pbcopy", "dictate on the terminal, then copy the text"),
    (
        "ffmpeg -loglevel error -i talk.mp4 -ar 16000 -ac 1 -f f32le - | dictate --input -",
        "transcribe a recording in any format, with word times",
    ),
    ("dictate --input fixtures/intros.fake.jsonl --json", "replay a fake script, to see the events"),
    ("dictate --hint obento --hints ~/.config/dictate/hints.txt", "favour some words"),
    ("dictate models pull", "get the model the config needs, once"),
];

const PROTOCOL: &str = "\
on a terminal (stdin a TTY, no --json): the words are drawn on the terminal as
  they are heard; Enter finishes and prints the text on stdout, Escape or
  Ctrl-C discards it. Anywhere else, or with --json, it speaks JSON lines:
events, one JSON object per line on stdout:
  loading, recording (device), level (level), preview (text), final (text,
  words), transcribing, text (text), no-speech, error (message); and from
  dictate models pull, downloading (model, bytes, total), as the download grows.
  level comes ten times a second while listening: how loud the input is,
  0 to 1 against the room's noise floor, for a level meter. words is each
  word of the final as {text, start, end}, in seconds from the input's start
commands, one per line on stdin: stop (finish and print the text), cancel
  (exit at once); the end of stdin, SIGINT and SIGTERM are cancel. With
  --input -, stdin is the audio instead: its end is the stop, and SIGINT and
  SIGTERM still cancel
exit codes: 0 text, 2 no speech, 3 cancelled, 1 error";

const CONFIG: &str = "\
config: $DICTATE_CONFIG, else ~/.config/dictate/config.toml; TOML, every key optional.
  A flag overrides its key; --hint and --hints add to hints. For example:
    mic = \"Yeti Stereo Microphone\"
    asr = \"nemo-speech:nemotron-en\"
    language = \"en-US\"
    timeout = 120
    hints = [\"obento\", \"Magit\", \"gptel\", \"Ghostty\"]";

const MODELS: &str = "\
models: kept in pam's model store, $PAM_MODELS, else
  ~/Library/Application Support/pam/models. The store is shared with pam: a
  model either pulled is there for both, and one removed is gone for both.
  dictate never downloads one itself: a missing model stops it before the mic
  opens, saying what to pull.
  dictate models list            every model the store knows: its size, whether
                                 it is pulled, and `used` for the config's asr
  dictate models pull [<name>…]  download models, checked; with no name, the
                                 config's asr, and with hints, the copy of its
                                 model that boosts them
  dictate models rm <name>…      remove models from the store
  dictate models path            print the store's folder";

/// What the command line asks for.
pub enum Parsed {
    Run(Flags),
    Help,
    Version,
}

/// The options given, in order, before the config file fills in the rest.
pub struct Flags(Vec<(&'static str, Option<String>)>);

/// Parses the arguments (without the program name). An `Err` is a usage
/// error's message.
pub fn parse(args: &[String]) -> Result<Parsed, String> {
    let mut options: Vec<(&'static str, Option<String>)> = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Parsed::Help),
            "-V" | "--version" => return Ok(Parsed::Version),
            a if !a.starts_with("--") => return Err(format!("unexpected `{a}`: dictate takes options only")),
            _ => {}
        }
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (arg.as_str(), None),
        };
        let Some(spec) = ARGS.iter().find(|a| a.name == name) else {
            let known: Vec<&str> = ARGS.iter().map(|a| a.name).collect();
            return Err(format!("unknown option {name}; dictate takes {}", known.join(", ")));
        };
        let value = match (spec.value, inline) {
            (true, Some(v)) => Some(v),
            (true, None) => match args.next() {
                Some(v) if !v.starts_with("--") => Some(v.clone()),
                _ => return Err(format!("{name} needs a value")),
            },
            (false, Some(_)) => return Err(format!("{name} is a switch and takes no value")),
            (false, None) => None,
        };
        options.push((spec.name, value));
    }
    Ok(Parsed::Run(Flags(options)))
}

impl Flags {
    /// Whether `--json` was given.
    pub fn json(&self) -> bool {
        self.0.iter().any(|(n, _)| *n == "--json")
    }

    /// The options to run with: these flags, and the config's keys for the
    /// rest. An `Err` is a usage error's message.
    pub fn resolve(&self, config: &Config) -> Result<Options, String> {
        let options = &self.0;
        let value = |name: &str| options.iter().rev().find(|(n, _)| *n == name).and_then(|(_, v)| v.clone());
        let input = match value("--input").as_deref() {
            None | Some("mic") => Input::Mic(value("--device").or_else(|| config.mic.clone())),
            Some(_) if value("--device").is_some() => return Err("--device is for --input mic".into()),
            Some("-") => Input::Stdin,
            Some(path) if path.ends_with(".fake.jsonl") => {
                if value("--asr").is_some() {
                    return Err("--asr doesn't go with a fake script: the script is the transcriber".into());
                }
                Input::Script(PathBuf::from(path))
            }
            Some(path) if path.ends_with(".wav") => Input::Wav(PathBuf::from(path)),
            Some(other) => {
                return Err(format!(
                    "--input {other}: give mic, - for stdin, a .wav recording or a .fake.jsonl script"
                ));
            }
        };
        let realtime = options.iter().any(|(n, _)| *n == "--realtime");
        if realtime && matches!(input, Input::Mic(_)) {
            return Err("--realtime is for stdin, a recording or a script; the mic is always live".into());
        }
        let seconds = match value("--timeout") {
            None => config.timeout,
            Some(t) => match t.parse::<f64>() {
                Ok(s) if s >= 0.0 && s.is_finite() => s,
                _ => return Err(format!("--timeout {t}: give a number of seconds, e.g. 30 or 2.5")),
            },
        };
        let timeout = (seconds > 0.0).then(|| (seconds * SAMPLE_RATE as f64).round() as Ts);
        let mut hints = config.hints.clone();
        hints.extend(options.iter().filter(|(n, _)| *n == "--hint").filter_map(|(_, v)| v.clone()));
        Ok(Options {
            input,
            asr: value("--asr").unwrap_or_else(|| config.asr.clone()),
            language: value("--language").unwrap_or_else(|| config.language.clone()),
            hints,
            hints_file: value("--hints").map(PathBuf::from),
            timeout,
            realtime,
        })
    }
}

/// The usage, options with their config keys, the config file, protocol
/// and examples: `--help`, and what every usage error shows.
pub fn manual() -> String {
    let mut s = format!(
        "dictate: listens, transcribes live, and hands back the text.\n\nusage: {USAGE}\n\n\
         options, beside the config file's key each overrides or adds to:\n"
    );
    let width = ARGS.iter().map(|a| a.name.len()).max().unwrap_or(0) + 2;
    let key_width = ARGS.iter().map(|a| a.key.len()).max().unwrap_or(0) + 2;
    for a in ARGS {
        writeln!(s, "  {:width$}{:key_width$}{}", a.name, a.key, a.help).unwrap();
    }
    writeln!(s, "\n{CONFIG}\n\n{MODELS}\n\n{PROTOCOL}\n\nexamples:").unwrap();
    let width = EXAMPLES.iter().map(|(e, _)| e.len()).max().unwrap_or(0) + 2;
    for (e, why) in EXAMPLES {
        writeln!(s, "  {e:width$} # {why}").unwrap();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(line: &str) -> Result<(Options, bool), String> {
        run_with(line, &Config::default())
    }

    fn run_with(line: &str, config: &Config) -> Result<(Options, bool), String> {
        let args: Vec<String> = line.split_whitespace().map(str::to_string).collect();
        match parse(&args)? {
            Parsed::Run(flags) => Ok((flags.resolve(config)?, flags.json())),
            _ => panic!("not a run: {line}"),
        }
    }

    #[test]
    fn the_config_file_fills_in_what_the_flags_leave_out() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dictate.toml");
        std::fs::write(
            &path,
            "mic = \"Yeti Stereo Microphone\"\nasr = \"qwen:qwen3-asr-1.7b\"\nlanguage = \"zh-CN\"\n\
             timeout = 120\nhints = [\"obento\", \"Magit\"]\n",
        )
        .unwrap();
        let config =
            Config::load_with(|name| (name == "DICTATE_CONFIG").then(|| path.clone().into())).unwrap();

        let (o, _) = run_with("", &config).unwrap();
        assert!(matches!(o.input, Input::Mic(Some(ref d)) if d == "Yeti Stereo Microphone"));
        assert_eq!((o.asr.as_str(), o.language.as_str()), ("qwen:qwen3-asr-1.7b", "zh-CN"));
        assert_eq!(
            (o.timeout, o.hints),
            (Some(120 * 16_000), vec!["obento".to_string(), "Magit".to_string()])
        );

        let (o, _) = run_with(
            "--device Blue --asr nemo-speech:x --language en-US --timeout 0 --hint gptel --hints h.txt",
            &config,
        )
        .unwrap();
        assert!(matches!(o.input, Input::Mic(Some(ref d)) if d == "Blue"));
        assert_eq!((o.asr.as_str(), o.language.as_str(), o.timeout), ("nemo-speech:x", "en-US", None));
        assert_eq!(o.hints, ["obento", "Magit", "gptel"], "--hint adds to the config's hints");
        assert_eq!(o.hints_file, Some(PathBuf::from("h.txt")));

        // The config's mic and transcriber are for the mic and a recording; a script brings its own.
        let (o, _) = run_with("--input a.fake.jsonl", &config).unwrap();
        assert!(matches!(o.input, Input::Script(_)));
        let (o, _) = run_with("--input a.wav", &config).unwrap();
        assert!(matches!(o.input, Input::Wav(_)));
        assert_eq!(o.asr, "qwen:qwen3-asr-1.7b");
    }

    #[test]
    fn help_names_the_config_file_and_each_key_beside_its_flag() {
        let help = manual();
        assert!(help.contains("$DICTATE_CONFIG, else ~/.config/dictate/config.toml"), "{help}");
        for (flag, key) in [
            ("--device", "mic"),
            ("--asr", "asr"),
            ("--language", "language"),
            ("--hint", "hints"),
            ("--timeout", "timeout"),
        ] {
            let line = help.lines().find(|l| l.trim_start().starts_with(&format!("{flag} "))).unwrap();
            assert_eq!(line.split_whitespace().nth(1), Some(key), "{line}");
        }
    }

    #[test]
    fn defaults_to_the_default_mic_and_nemotron() {
        let (o, json) = run("").unwrap();
        assert!(matches!(o.input, Input::Mic(None)));
        assert_eq!(
            (o.asr.as_str(), o.language.as_str(), o.timeout, json),
            ("nemo-speech:nemotron-en", "en-US", None, false)
        );
    }

    #[test]
    fn reads_every_option() {
        let (o, json) =
            run("--device=Yeti --asr qwen:x --language auto --hint Pam --hint=obento --hints h.txt --timeout 2.5 --json")
                .unwrap();
        assert!(matches!(o.input, Input::Mic(Some(ref d)) if d == "Yeti"));
        assert_eq!((o.asr.as_str(), o.language.as_str()), ("qwen:x", "auto"));
        assert_eq!(o.hints, ["Pam", "obento"]);
        assert_eq!(o.hints_file, Some(PathBuf::from("h.txt")));
        assert_eq!((o.timeout, json), (Some(40_000), true));
        assert!(!o.realtime);
        assert_eq!(run("--timeout 0").unwrap().0.timeout, None);
    }

    #[test]
    fn the_input_is_chosen_by_its_name() {
        assert!(matches!(run("--input a.wav").unwrap().0.input, Input::Wav(_)));
        assert!(matches!(run("--input a.fake.jsonl").unwrap().0.input, Input::Script(_)));
        assert!(matches!(run("--input -").unwrap().0.input, Input::Stdin));
        assert!(run("--input - --realtime").unwrap().0.realtime);
        assert!(run("--input a.mp3").unwrap_err().contains(".wav"));
        assert!(run("--input a.wav --device Yeti").unwrap_err().contains("--input mic"));
        assert!(run("--input a.fake.jsonl --asr fake:b.jsonl").is_err());
        assert!(run("--input a.fake.jsonl --realtime").unwrap().0.realtime);
        assert!(run("--realtime").unwrap_err().contains("always live"));
    }

    #[test]
    fn usage_errors_say_what_is_wrong() {
        assert_eq!(run("--timeout").unwrap_err(), "--timeout needs a value");
        assert!(run("--timeout soon").unwrap_err().contains("seconds"));
        assert!(run("--timeout -1").is_err());
        assert!(run("--json=1").unwrap_err().contains("switch"));
        assert!(run("--bogus").unwrap_err().contains("--input"));
        assert!(run("stop").unwrap_err().contains("options only"));
        assert!(matches!(parse(&["--help".into()]), Ok(Parsed::Help)));
    }
}
