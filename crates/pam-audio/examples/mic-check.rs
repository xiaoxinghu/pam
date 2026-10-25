//! Listens to a mic the way a meeting does, and prints its level every
//! 100 ms: to check the mic works and has permission, and to see where speech
//! and room noise fall on the level scale.
//!
//! ```text
//! cargo run --release -p pam-audio --example mic-check -- [--device <name>] [--secs 10] [--wav out.wav]
//! ```
//!
//! It listens to the input named by `--device`, such as the `mic` in Pam's
//! config file, or the default input.
//!
//! `--wav` keeps what was heard, as 16 kHz mono, exactly as a meeting would
//! record it to `audio.wav`.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result, bail};
use pam_audio::{self as audio, GATE_DB, Meter, Mic, RANGE_DB, Recorder};
use pam_speech::{SAMPLE_RATE, Ts};

/// Samples in one second.
const SECOND: Ts = SAMPLE_RATE as Ts;
/// What a meeting reads from the mic at a time: 100 ms.
const CHUNK: usize = 1_600;

const USAGE: &str = "usage: mic-check [--device <name>] [--secs 10] [--wav out.wav]";

fn main() -> Result<()> {
    let mut device = None;
    let mut secs = 10.0;
    let mut wav = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--device" => device = Some(args.next().context("--device needs an input's name")?),
            "--secs" => secs = args.next().context("--secs needs a number")?.parse()?,
            "--wav" => wav = Some(PathBuf::from(args.next().context("--wav needs a path")?)),
            other => bail!("unknown argument `{other}`; {USAGE}"),
        }
    }
    if !(secs > 0.0 && secs < f64::INFINITY) {
        bail!("--secs must be a positive number of seconds, not {secs}");
    }
    let mut mic = Mic::open(device.as_deref())?;
    println!(
        "{}; level 0 up to {GATE_DB} dB above the noise floor, 1 at {} dB above",
        mic.describe(),
        GATE_DB + RANGE_DB
    );
    let mut meter = Meter::default();
    let mut rec = wav.as_deref().map(Recorder::create).transpose()?;
    let stop = AtomicBool::new(false);
    mic.start()?;
    let total = (secs * SECOND as f64) as u64;
    let mut heard = 0;
    let mut levels = Vec::new();
    while heard < total {
        let chunk = mic.read(CHUNK, &stop)?;
        heard += chunk.len() as u64;
        if let Some(rec) = &mut rec {
            rec.write(&chunk)?;
        }
        for w in mic.warnings() {
            println!("warning: {w}");
        }
        let db = audio::dbfs(&chunk).unwrap_or(f32::NEG_INFINITY);
        let level = meter.level(&chunk);
        levels.push(level);
        let bar = "█".repeat((level * 40.0).round() as usize);
        println!("{:6.1}s {db:6.1} dBFS {level:4.2} {bar}", heard as f64 / SECOND as f64);
    }
    if let Some(rec) = rec {
        rec.finish()?;
    }
    levels.sort_by(f32::total_cmp);
    let q = |p: f64| levels[((levels.len() - 1) as f64 * p) as usize];
    println!("level p10 {:.2}, p50 {:.2}, p90 {:.2}, max {:.2}", q(0.1), q(0.5), q(0.9), q(1.0));
    Ok(())
}
