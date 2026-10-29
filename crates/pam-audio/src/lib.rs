//! Audio in and out: the microphone, a WAV file to replay, recording
//! `audio.wav`, and how loud the input is.
//!
//! Everything inside the core is 16 kHz mono f32.
//! The mic is converted to that once, here, before anything else sees it.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Read};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, unbounded};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use pam_speech::{SAMPLE_RATE, Ts};

/// Samples in one second.
const SECOND: Ts = SAMPLE_RATE as Ts;

/// How much captured audio may wait for the meeting. Past this, the backends
/// have fallen this far behind, and new audio is dropped until they catch up.
const BACKLOG: Ts = 30 * SECOND;
/// How long the mic may send nothing but zeros before Pam says why.
const ZEROS: Ts = 2 * SECOND;
/// How long a running mic may send nothing before it counts as gone. CoreAudio
/// delivers buffers continuously, zeros included, so silence from the stream
/// itself means it stopped.
const STALL: Duration = Duration::from_secs(5);
/// A chunk this far above the noise floor starts to count as sound; see [`Meter`].
pub const GATE_DB: f32 = 6.0;
/// The rise above the gate that reads as full level.
pub const RANGE_DB: f32 = 24.0;
/// How fast the noise floor creeps up while the input stays loud, per second
/// of audio. It drops at once to any quieter chunk.
pub const FLOOR_RISE_DB: f32 = 0.5;

/// The RMS level of `chunk` in dB below full scale; `None` for pure zeros.
pub fn dbfs(chunk: &[f32]) -> Option<f32> {
    let power = chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len().max(1) as f32;
    (power > 0.0).then(|| 10.0 * power.log10())
}

/// How loud the input is, from 0 to 1, against the room's own noise floor.
///
/// Mics and rooms differ by tens of dB: through one USB interface, speech
/// from across the room came in at -70 dBFS over a -84 dBFS room. So the
/// level is the rise above the quietest recent chunk: 0 up to [`GATE_DB`]
/// above it, 1 at [`GATE_DB`] + [`RANGE_DB`].
#[derive(Debug, Default)]
pub struct Meter {
    floor: Option<f32>,
}

impl Meter {
    pub fn level(&mut self, chunk: &[f32]) -> f32 {
        let Some(db) = dbfs(chunk) else { return 0.0 };
        let rise = FLOOR_RISE_DB * chunk.len() as f32 / SAMPLE_RATE as f32;
        let floor = self.floor.map_or(db, |f| db.min(f + rise));
        self.floor = Some(floor);
        ((db - floor - GATE_DB) / RANGE_DB).clamp(0.0, 1.0)
    }
}

/// Converts mono audio at a device's rate to 16 kHz, in pieces of any size.
pub struct To16k {
    /// `None` when the device already runs at 16 kHz.
    fft: Option<Fft<f32>>,
    /// Input the resampler hasn't taken yet.
    pending: Vec<f32>,
    out: Vec<f32>,
    /// Output still to drop: the resampler's delay, so the output lines up with the input.
    skip: usize,
}

impl To16k {
    pub fn new(rate: u32) -> Result<To16k> {
        if rate == SAMPLE_RATE {
            return Ok(To16k { fft: None, pending: Vec::new(), out: Vec::new(), skip: 0 });
        }
        let fft = Fft::new(rate as usize, SAMPLE_RATE as usize, 1_024, 1, FixedSync::Input)
            .map_err(|e| anyhow!("can't resample {rate} Hz to 16 kHz: {e}"))?;
        let out = vec![0.0; fft.output_frames_max()];
        let skip = fft.output_delay();
        Ok(To16k { fft: Some(fft), pending: Vec::new(), out, skip })
    }

    /// Takes mono samples at the device rate and returns every 16 kHz sample
    /// they complete.
    pub fn push(&mut self, mono: &[f32]) -> Vec<f32> {
        let Some(fft) = &mut self.fft else { return mono.to_vec() };
        self.pending.extend_from_slice(mono);
        let mut done = Vec::new();
        let mut used = 0;
        while self.pending.len() - used >= fft.input_frames_next() {
            let input = &self.pending[used..];
            let input = InterleavedSlice::new(input, 1, input.len()).expect("one channel");
            let frames = self.out.len();
            let mut output = InterleavedSlice::new_mut(&mut self.out, 1, frames).expect("one channel");
            let (read, wrote) =
                fft.process_into_buffer(&input, &mut output, None).expect("buffers sized by the resampler");
            used += read;
            let skipped = wrote.min(self.skip);
            self.skip -= skipped;
            done.extend_from_slice(&self.out[skipped..wrote]);
        }
        self.pending.drain(..used);
        done
    }
}

/// A microphone, as 16 kHz mono.
///
/// The capture callback only downmixes; resampling happens in [`Mic::read`],
/// on the reading thread. Nothing is captured until [`Mic::start`].
pub struct Mic {
    stream: cpal::Stream,
    /// Mono audio at the device rate, from the callback.
    audio: Receiver<Vec<f32>>,
    /// Device frames waiting in `audio`.
    queued: Arc<AtomicUsize>,
    /// Device frames the callback dropped because too many were waiting.
    dropped: Arc<AtomicUsize>,
    errors: Receiver<cpal::Error>,
    /// Stream errors that didn't stop it, for [`Mic::warnings`].
    noted: Vec<String>,
    convert: To16k,
    ready: VecDeque<f32>,
    /// The device's name, e.g. "MacBook Pro Microphone".
    pub name: String,
    /// The device's sample rate.
    pub rate: u32,
    channels: u16,
    /// Samples read so far, and whether every one was exactly zero.
    read: Ts,
    all_zero: bool,
}

impl Mic {
    /// Opens the input device called `name` (any case), or the system's
    /// default input, in its own format.
    pub fn open(name: Option<&str>) -> Result<Mic> {
        let host = cpal::default_host();
        let device = match name {
            None => host
                .default_input_device()
                .context("no microphone found: connect one, or replay a recording with --input <file.wav>")?,
            Some(want) => {
                let inputs: Vec<(String, cpal::Device)> = host
                    .input_devices()
                    .context("listing the input devices")?
                    .map(|d| (device_name(&d), d))
                    .collect();
                let names: Vec<&str> = inputs.iter().map(|(n, _)| n.as_str()).collect();
                let names = names.join(", ");
                inputs
                    .into_iter()
                    .find(|(n, _)| n.eq_ignore_ascii_case(want))
                    .map(|(_, d)| d)
                    .with_context(|| format!("no input device called \"{want}\"; the inputs are: {names}"))?
            }
        };
        let name = device_name(&device);
        let config = device.default_input_config().with_context(|| format!("reading {name}'s settings"))?;
        let (rate, channels) = (config.sample_rate(), config.channels());
        let (audio_tx, audio) = unbounded();
        let (error_tx, errors) = unbounded();
        let capture = Capture {
            channels: channels as usize,
            backlog: (BACKLOG * rate as Ts / SAMPLE_RATE as Ts) as usize,
            tx: audio_tx,
            queued: Arc::default(),
            dropped: Arc::default(),
        };
        let (queued, dropped) = (capture.queued.clone(), capture.dropped.clone());
        let stream = match config.sample_format() {
            cpal::SampleFormat::I16 => capture.build::<i16>(&device, config.into(), error_tx),
            cpal::SampleFormat::I32 => capture.build::<i32>(&device, config.into(), error_tx),
            cpal::SampleFormat::F32 => capture.build::<f32>(&device, config.into(), error_tx),
            cpal::SampleFormat::F64 => capture.build::<f64>(&device, config.into(), error_tx),
            other => bail!("{name} sends {other} samples, which Pam can't read yet"),
        }
        .with_context(|| format!("opening {name}"))?;
        Ok(Mic {
            stream,
            audio,
            queued,
            dropped,
            errors,
            noted: Vec::new(),
            convert: To16k::new(rate)?,
            ready: VecDeque::new(),
            name,
            rate,
            channels,
            read: 0,
            all_zero: true,
        })
    }

    /// Starts capturing.
    pub fn start(&self) -> Result<()> {
        self.stream.play().with_context(|| format!("starting {}", self.name))
    }

    /// Up to `n` samples at 16 kHz. Waits for them, unless `stop` is set:
    /// then returns what is there, and empty means the end. Fails if the
    /// stream ends: the device goes away, its format changes, or it sends
    /// nothing for [`STALL`].
    pub fn read(&mut self, n: usize, stop: &AtomicBool) -> Result<Vec<f32>> {
        let mut heard = Instant::now();
        while self.ready.len() < n && !stop.load(Ordering::Relaxed) {
            for e in self.errors.try_iter() {
                if fatal(&e) {
                    bail!("{} stopped: {e}", self.name);
                }
                self.noted.push(format!("{}: {e}", self.name));
            }
            match self.audio.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => {
                    heard = Instant::now();
                    self.queued.fetch_sub(chunk.len(), Ordering::Relaxed);
                    self.ready.extend(self.convert.push(&chunk));
                }
                Err(RecvTimeoutError::Timeout) if heard.elapsed() >= STALL => {
                    bail!("{} sent no audio for {} s", self.name, STALL.as_secs())
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => bail!("{} stopped sending audio", self.name),
            }
        }
        let n = n.min(self.ready.len());
        let out: Vec<f32> = self.ready.drain(..n).collect();
        self.read += n as Ts;
        self.all_zero &= out.iter().all(|s| *s == 0.0);
        Ok(out)
    }

    /// Problems that don't stop the input, each reported once: dropped audio,
    /// a mic that only sends zeros, and stream errors that aren't fatal.
    pub fn warnings(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        let dropped = self.dropped.swap(0, Ordering::Relaxed);
        if dropped > 0 {
            out.push(format!(
                "the speech backends fell more than {}s behind the mic: {:.1}s of audio was dropped",
                BACKLOG / SECOND,
                dropped as f64 / self.rate as f64
            ));
        }
        out.extend(self.silence());
        out.append(&mut self.noted);
        out
    }

    /// Once the mic has sent nothing but zeros for [`ZEROS`], why that is
    /// likely; reported once, here or by [`Mic::warnings`].
    pub fn silence(&mut self) -> Option<String> {
        if !(self.all_zero && self.read >= ZEROS) {
            return None;
        }
        self.all_zero = false;
        Some(format!(
            "{} sends only silence: if this terminal has no microphone access, allow it in \
             System Settings → Privacy & Security → Microphone, then start again",
            self.name
        ))
    }

    /// e.g. "MacBook Pro Microphone (48000 Hz, 1 channel)".
    pub fn describe(&self) -> String {
        let s = if self.channels == 1 { "" } else { "s" };
        format!("{} ({} Hz, {} channel{s})", self.name, self.rate, self.channels)
    }
}

fn device_name(device: &cpal::Device) -> String {
    device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "microphone".into())
}

/// Errors after which the stream sends nothing more, or nothing Pam can read:
/// `StreamInvalidated` comes with a change of sample rate, which the resampler
/// was built for.
fn fatal(e: &cpal::Error) -> bool {
    matches!(
        e.kind(),
        cpal::ErrorKind::DeviceNotAvailable
            | cpal::ErrorKind::PermissionDenied
            | cpal::ErrorKind::HostUnavailable
            | cpal::ErrorKind::StreamInvalidated
    )
}

/// What the capture callback needs: it downmixes each buffer and queues it,
/// dropping it instead when the backlog is full. It never blocks.
struct Capture {
    channels: usize,
    /// The most device frames that may wait.
    backlog: usize,
    tx: Sender<Vec<f32>>,
    queued: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
}

impl Capture {
    fn build<T>(
        self,
        device: &cpal::Device,
        config: cpal::StreamConfig,
        errors: Sender<cpal::Error>,
    ) -> Result<cpal::Stream, cpal::Error>
    where
        T: cpal::SizedSample,
        f32: cpal::FromSample<T>,
    {
        device.build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let frames = data.len() / self.channels;
                if self.queued.load(Ordering::Relaxed) + frames > self.backlog {
                    self.dropped.fetch_add(frames, Ordering::Relaxed);
                    return;
                }
                let scale = 1.0 / self.channels as f32;
                let mono = data
                    .chunks_exact(self.channels)
                    .map(|f| f.iter().map(|s| cpal::Sample::to_sample::<f32>(*s)).sum::<f32>() * scale)
                    .collect();
                self.queued.fetch_add(frames, Ordering::Relaxed);
                let _ = self.tx.send(mono);
            },
            move |e| {
                let _ = errors.send(e);
            },
            None,
        )
    }
}

/// A WAV file read in chunks, downmixed to mono.
pub struct WavSource {
    samples: Box<dyn Iterator<Item = hound::Result<f32>> + Send>,
    channels: usize,
    /// Length in samples.
    pub duration: Ts,
}

impl WavSource {
    /// Opens a 16 kHz WAV file of any channel count and sample format.
    pub fn open(path: &Path) -> Result<WavSource> {
        let reader = hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
        let spec = reader.spec();
        if spec.sample_rate != SAMPLE_RATE {
            bail!(
                "{} is {} Hz; Pam replays 16 kHz audio only, e.g. `sox in.wav -r 16000 out.wav` first",
                path.display(),
                spec.sample_rate
            );
        }
        let duration = reader.duration() as Ts;
        let samples: Box<dyn Iterator<Item = hound::Result<f32>> + Send> = match spec.sample_format {
            hound::SampleFormat::Float => Box::new(reader.into_samples::<f32>()),
            hound::SampleFormat::Int => {
                let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
                Box::new(reader.into_samples::<i32>().map(move |s| s.map(|s| s as f32 * scale)))
            }
        };
        Ok(WavSource { samples, channels: spec.channels as usize, duration })
    }

    /// Up to `n` mono samples; fewer only at the end of the file.
    pub fn read(&mut self, n: usize) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(n);
        'frames: while out.len() < n {
            let mut sum = 0.0;
            for c in 0..self.channels {
                match self.samples.next() {
                    Some(s) => sum += s.context("reading the input WAV")?,
                    None if c == 0 => break 'frames,
                    None => bail!("the input WAV ends in the middle of a frame"),
                }
            }
            out.push(sum / self.channels as f32);
        }
        Ok(out)
    }
}

/// Raw 16 kHz mono 32-bit float samples, little-endian, from a stream such as
/// stdin: what `ffmpeg -ar 16000 -ac 1 -f f32le -` writes. Raw, not a WAV:
/// a WAV written to a pipe can't say its length in its header.
pub struct RawSource {
    stream: Box<dyn Read + Send>,
    /// The bytes of a sample split between two reads.
    partial: Vec<u8>,
}

impl RawSource {
    pub fn new(stream: impl Read + Send + 'static) -> RawSource {
        RawSource { stream: Box::new(stream), partial: Vec::new() }
    }

    /// Up to `n` samples, waiting for them; fewer only at the end of the stream.
    pub fn read(&mut self, n: usize) -> Result<Vec<f32>> {
        let mut bytes = std::mem::take(&mut self.partial);
        let want = n * 4;
        let mut buf = [0u8; 16_384];
        while bytes.len() < want {
            let len = buf.len().min(want - bytes.len());
            match self.stream.read(&mut buf[..len]) {
                Ok(0) => break,
                Ok(got) => bytes.extend_from_slice(&buf[..got]),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e).context("reading the raw audio"),
            }
        }
        let whole = bytes.len() / 4 * 4;
        if bytes.len() < want && whole < bytes.len() {
            bail!("the raw audio ends in the middle of a sample: is it 32-bit float (f32le)?");
        }
        self.partial = bytes.split_off(whole);
        Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }
}

/// An open input: the mic, a 16 kHz WAV, raw samples from a stream, or
/// silence of a given length, which drives the `fake` speech backends,
/// which only count samples.
pub enum Source {
    Mic(Box<Mic>),
    Wav(WavSource),
    Raw(RawSource),
    Silence { left: Ts },
}

impl Source {
    pub fn start(&self) -> Result<()> {
        match self {
            Source::Mic(mic) => mic.start(),
            _ => Ok(()),
        }
    }

    /// The next chunk of up to `n` samples; empty at the end. The mic ends
    /// once `stop` is set.
    pub fn read(&mut self, n: usize, stop: &AtomicBool) -> Result<Vec<f32>> {
        match self {
            Source::Mic(mic) => mic.read(n, stop),
            Source::Wav(wav) => wav.read(n),
            Source::Raw(raw) => raw.read(n),
            Source::Silence { left } => {
                let n = (*left).min(n as Ts);
                *left -= n;
                Ok(vec![0.0; n as usize])
            }
        }
    }

    /// Problems worth reporting that don't end the input: the mic's.
    pub fn warnings(&mut self) -> Vec<String> {
        match self {
            Source::Mic(mic) => mic.warnings(),
            _ => Vec::new(),
        }
    }

    /// Real audio has a level and is worth recording; the fake backends'
    /// silence has neither.
    pub fn is_audio(&self) -> bool {
        !matches!(self, Source::Silence { .. })
    }
}

/// Holds a replayed recording or script to a steady pace on an absolute
/// clock, so sleeps never add up to drift: real time at `speed` 1, or that
/// many times faster.
pub struct Pacer {
    start: Instant,
    speed: f64,
}

impl Pacer {
    pub fn start(speed: f64) -> Pacer {
        Pacer { start: Instant::now(), speed }
    }

    /// Sleeps until `ts` worth of audio has played since the start.
    pub fn wait_until(&self, ts: Ts) {
        let due = self.start + Duration::from_secs_f64(ts as f64 / SAMPLE_RATE as f64 / self.speed);
        if let Some(left) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(left);
        }
    }
}

/// Records what the meeting heard as `audio.wav`: 16 kHz mono, 16-bit.
pub struct Recorder {
    wav: hound::WavWriter<BufWriter<File>>,
}

impl Recorder {
    pub fn create(path: &Path) -> Result<Recorder> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let wav =
            hound::WavWriter::create(path, spec).with_context(|| format!("creating {}", path.display()))?;
        Ok(Recorder { wav })
    }

    pub fn write(&mut self, audio: &[f32]) -> Result<()> {
        for s in audio {
            self.wav.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
        }
        Ok(())
    }

    /// Writes buffered audio and updates the header, so the file is whole
    /// even if Pam stops without `finish`.
    pub fn flush(&mut self) -> Result<()> {
        self.wav.flush().context("writing audio.wav")
    }

    pub fn finish(self) -> Result<()> {
        self.wav.finalize().context("writing audio.wav")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recording_reads_back_as_the_same_mono_audio() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audio.wav");
        let audio: Vec<f32> = (0..4_000).map(|i| ((i as f32) / 50.0).sin() * 0.5).collect();
        let mut rec = Recorder::create(&path).unwrap();
        rec.write(&audio[..1_000]).unwrap();
        rec.flush().unwrap();
        rec.write(&audio[1_000..]).unwrap();
        rec.finish().unwrap();

        let mut src = WavSource::open(&path).unwrap();
        assert_eq!(src.duration, 4_000);
        let back = [src.read(3_000).unwrap(), src.read(3_000).unwrap()].concat();
        assert_eq!(back.len(), 4_000);
        assert!(back.iter().zip(&audio).all(|(a, b)| (a - b).abs() < 1e-4));
        assert!(src.read(10).unwrap().is_empty());
    }

    /// A stream that hands out at most `step` bytes a read, as a pipe may.
    struct Trickle {
        bytes: Vec<u8>,
        at: usize,
        step: usize,
    }

    impl Read for Trickle {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = buf.len().min(self.step).min(self.bytes.len() - self.at);
            buf[..n].copy_from_slice(&self.bytes[self.at..self.at + n]);
            self.at += n;
            Ok(n)
        }
    }

    fn f32le(audio: &[f32]) -> Vec<u8> {
        audio.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn raw_samples_read_whole_however_the_stream_splits_them() {
        let audio: Vec<f32> = (0..1_000).map(|i| ((i as f32) / 50.0).sin() * 0.5).collect();
        for step in [1, 3, 4, 4_001, 1 << 20] {
            let mut src = RawSource::new(Trickle { bytes: f32le(&audio), at: 0, step });
            let back = [src.read(600).unwrap(), src.read(600).unwrap()].concat();
            assert_eq!(back, audio, "{step} bytes a read");
            assert!(src.read(10).unwrap().is_empty());
        }

        let mut cut = f32le(&[0.25, 0.5]);
        cut.pop();
        let mut src = RawSource::new(Trickle { bytes: cut, at: 0, step: 3 });
        assert_eq!(src.read(1).unwrap(), [0.25]);
        let err = src.read(10).unwrap_err().to_string();
        assert!(err.contains("middle of a sample") && err.contains("f32le"), "{err}");
    }

    fn tone(rate: u32, hz: f32, secs: f32, amp: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        (0..n).map(|i| (std::f32::consts::TAU * hz * i as f32 / rate as f32).sin() * amp).collect()
    }

    #[test]
    fn mic_rates_are_resampled_to_16k_in_pieces_of_any_size() {
        for rate in [48_000, 44_100, 16_000] {
            let input = tone(rate, 440.0, 2.0, 0.5);
            let mut conv = To16k::new(rate).unwrap();
            let mut out = Vec::new();
            for piece in input.chunks(479) {
                out.extend(conv.push(piece));
            }
            // All of it but what the resampler still holds (under one chunk).
            assert!(out.len() <= 32_000 && out.len() > 32_000 - 1_000, "{rate}: {}", out.len());
            // The same tone, in phase: the resampler's delay is trimmed. From
            // 44.1 kHz the delay isn't a whole sample, which leaves ~0.2 of one.
            let want = tone(16_000, 440.0, 2.0, 0.5);
            let err = out[100..].iter().zip(&want[100..]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(err < 0.02, "{rate}: off by {err}");
        }
    }

    #[test]
    fn resampling_removes_what_16k_cannot_hold() {
        // A 10 kHz tone is above 16 kHz's Nyquist limit: it must not fold back as 6 kHz.
        let mut conv = To16k::new(48_000).unwrap();
        let out = conv.push(&tone(48_000, 10_000.0, 1.0, 0.5));
        assert!(dbfs(&out[1_000..]).unwrap() < -60.0, "{:?}", dbfs(&out[1_000..]));
    }

    #[test]
    fn the_meter_reads_sound_against_the_rooms_noise_floor() {
        let mut m = Meter::default();
        let quiet = tone(16_000, 300.0, 0.1, 0.000_1); // -83 dBFS
        let speech = tone(16_000, 300.0, 0.1, 0.000_8); // -65 dBFS: 18 dB over the room
        let loud = tone(16_000, 300.0, 0.1, 0.1); // -23 dBFS
        assert_eq!(m.level(&quiet), 0.0);
        assert_eq!(m.level(&quiet), 0.0);
        let l = m.level(&speech);
        assert!((l - 0.5).abs() < 0.03, "{l}");
        assert_eq!(m.level(&loud), 1.0);
        // Pure zeros (a muted or blocked mic) are silence and leave the floor alone.
        assert_eq!(m.level(&[0.0; 1_600]), 0.0);
        assert!((m.level(&speech) - l).abs() < 0.01);
        // Ten seconds of steady sound raise the floor by 5 dB.
        let before = m.level(&speech);
        for _ in 0..100 {
            m.level(&speech);
        }
        assert!(before - m.level(&speech) > 4.0 / RANGE_DB, "{before}");
    }

    #[test]
    fn a_stream_that_must_be_rebuilt_ends_the_mic() {
        let kind = |k| fatal(&cpal::Error::from(k));
        assert!(kind(cpal::ErrorKind::StreamInvalidated));
        assert!(kind(cpal::ErrorKind::DeviceNotAvailable));
        assert!(!kind(cpal::ErrorKind::Xrun));
        assert!(!kind(cpal::ErrorKind::DeviceChanged));
    }

    #[test]
    fn stereo_is_downmixed_and_other_rates_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, rate: u32, channels: u16, samples: &[i16]| {
            let path = dir.path().join(name);
            let spec = hound::WavSpec {
                channels,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(&path, spec).unwrap();
            for s in samples {
                w.write_sample(*s).unwrap();
            }
            w.finalize().unwrap();
            path
        };
        let stereo = write("stereo.wav", 16_000, 2, &[16_384, 0, -16_384, -16_384]);
        let mut src = WavSource::open(&stereo).unwrap();
        assert_eq!(src.duration, 2);
        assert_eq!(src.read(10).unwrap(), [0.25, -0.5]);

        let cd = write("cd.wav", 44_100, 1, &[0; 10]);
        let err = WavSource::open(&cd).err().unwrap().to_string();
        assert!(err.contains("44100 Hz") && err.contains("16 kHz"), "{err}");
    }
}
