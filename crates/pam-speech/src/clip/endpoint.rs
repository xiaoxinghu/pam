//! The shared endpointer: cuts a stream into clips, and says why
//! it cut. It is an energy gate measured against the room: a frame is speech
//! when it is well above the room's noise floor, and never when it is below
//! a fixed level, so digital silence and a quiet room are judged as the live
//! prototype's gate judged them. Every clip model is cut by it, so replacing
//! it changes them all at once. A stream adapter whose engine has no
//! endpointing of its own uses it too.

use crate::{SAMPLE_RATE, Ts};

/// Frames the gate looks at: 20 ms.
pub const FRAME: Ts = SAMPLE_RATE as Ts / 50;
/// No frame quieter than this is speech: −60 dBFS RMS. The prototype first
/// tried −40 dBFS, which lost quiet AMI speech.
pub const THRESHOLD_DB: f64 = -60.0;
/// How far above the room's noise floor a frame must be to be speech: 10 dB.
/// On the Yeti in a room at −45 dBFS, 20 ms frames of the room alone ranged
/// from −52 to −41 dBFS around a floor of −46.
const ABOVE_ROOM_DB: f64 = 10.0;
/// How fast the noise floor rises while the room stays louder than it, per
/// second of audio. It drops at once to any quieter level.
pub const FLOOR_RISE_DB: f64 = 0.5;
/// The share of each frame's power in the level the floor follows: the room
/// is heard over about 100 ms, so one quiet frame doesn't drag the floor down.
const SMOOTHING: f64 = 0.2;
/// A clip ends after this much audio that is not speech: 800 ms, as NeMo's
/// endpointing.
pub const END_SILENCE: Ts = SAMPLE_RATE as Ts * 4 / 5;
/// Audio before the first speech frame that a clip starts with: 200 ms, or
/// less when the last clip ended closer than that.
pub const PRE_ROLL: Ts = SAMPLE_RATE as Ts / 5;
/// The longest clip: 20 s. Speech that goes on is cut here, and goes on in
/// the next clip.
pub const CAP: Ts = 20 * SAMPLE_RATE as Ts;

/// Why a clip ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cut {
    /// After [`END_SILENCE`] that is not speech: the utterance ended.
    Silence,
    /// At [`CAP`], perhaps with speech going on: the next clip continues
    /// the same utterance if its speech starts within [`END_SILENCE`] of the
    /// last.
    Cap,
    /// At the end of the stream.
    End,
}

/// A finished clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    /// Where it starts on the stream's clock.
    pub start: Ts,
    pub audio: Vec<f32>,
    pub cut: Cut,
    /// Whether it starts an utterance: always, unless the last clip was cut
    /// at the cap and speech went on within [`END_SILENCE`]. A silence that
    /// begins just before the cap still ends the utterance.
    pub starts_utterance: bool,
}

impl Clip {
    pub fn end(&self) -> Ts {
        self.start + self.audio.len() as Ts
    }
}

/// The room's noise floor, which a frame must rise [`ABOVE_ROOM_DB`] above
/// to be speech.
///
/// Mics and rooms differ by tens of dB: the Yeti in a quiet office hears the
/// room at −45 dBFS, where a fixed −60 dBFS gate takes every frame for speech
/// and no clip ends on a silence. So the floor is the quietest recent level,
/// as `pam_audio::Meter`'s is, heard over about 100 ms. It starts at the
/// first frame, which is never speech, and never sinks below
/// [`THRESHOLD_DB`] − [`ABOVE_ROOM_DB`], so digital silence and a quiet room
/// leave the fixed threshold in charge.
#[derive(Debug, Default)]
struct Room {
    /// Frame power, smoothed.
    level: Option<f64>,
    /// In dBFS.
    floor: Option<f64>,
}

impl Room {
    /// Hears a frame of `len` samples whose mean power is `power`, and says
    /// whether it is speech.
    fn speech(&mut self, power: f64, len: Ts) -> bool {
        let level = self.level.map_or(power, |l| l + (power - l) * SMOOTHING);
        self.level = Some(level);
        let rise = FLOOR_RISE_DB * len as f64 / f64::from(SAMPLE_RATE);
        let floor =
            self.floor.map_or(dbfs(level), |f| dbfs(level).min(f + rise)).max(THRESHOLD_DB - ABOVE_ROOM_DB);
        self.floor = Some(floor);
        dbfs(power) >= floor + ABOVE_ROOM_DB
    }
}

/// A mean power in dB below full scale; −∞ for silence.
fn dbfs(power: f64) -> f64 {
    10.0 * power.log10()
}

/// Cuts pushed audio into clips, 20 ms frame by frame. Audio between clips,
/// which is not speech, belongs to none, apart from the pre-roll.
#[derive(Debug, Default)]
pub struct Endpointer {
    /// What the frames looked at say about the room.
    room: Room,
    /// The audio kept: the open clip, or what would be its pre-roll, then
    /// any part of a frame not looked at yet.
    audio: Vec<f32>,
    /// Where `audio[0]` is on the stream's clock.
    kept: Ts,
    /// Where the frames looked at end.
    heard: Ts,
    /// Where the open clip starts, when one is open.
    open: Option<Ts>,
    /// Where the last speech frame ended.
    last_speech: Ts,
    /// Where the last clip ended.
    last_cut: Ts,
    /// Whether the last clip was cut at the cap.
    capped: bool,
    /// Whether the open clip starts an utterance.
    starts_utterance: bool,
}

impl Endpointer {
    pub fn new() -> Endpointer {
        Endpointer::default()
    }

    /// Takes more audio, and returns the clips it closed.
    pub fn push(&mut self, audio: &[f32]) -> Vec<Clip> {
        self.audio.extend_from_slice(audio);
        let mut clips = Vec::new();
        while self.pending() >= FRAME {
            clips.extend(self.frame(FRAME));
        }
        clips
    }

    /// Ends the stream: looks at the last part of a frame, and closes the
    /// open clip there. Returns the clips closed.
    pub fn finish(&mut self) -> Vec<Clip> {
        let mut clips = Vec::new();
        if self.pending() > 0 {
            clips.extend(self.frame(self.pending()));
        }
        clips.extend(self.cut(Cut::End));
        clips
    }

    /// The open clip so far, and where it starts: up to the frames looked at.
    pub fn open(&self) -> Option<(Ts, &[f32])> {
        let start = self.open?;
        Some((start, &self.audio[(start - self.kept) as usize..(self.heard - self.kept) as usize]))
    }

    /// Whether the open clip starts an utterance (see [`Clip::starts_utterance`]).
    pub fn open_starts_utterance(&self) -> bool {
        self.open.is_some() && self.starts_utterance
    }

    /// Where the audio looked at ends.
    pub fn heard(&self) -> Ts {
        self.heard
    }

    /// Audio pushed that no frame has looked at yet.
    fn pending(&self) -> Ts {
        self.kept + self.audio.len() as Ts - self.heard
    }

    /// Looks at the next `len` samples.
    fn frame(&mut self, len: Ts) -> Option<Clip> {
        let from = (self.heard - self.kept) as usize;
        let frame = &self.audio[from..from + len as usize];
        let power = frame.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / len as f64;
        let start = self.heard;
        self.heard += len;
        if self.room.speech(power, len) {
            if self.open.is_none() {
                self.open = Some(start.saturating_sub(PRE_ROLL).max(self.last_cut).max(self.kept));
                // After a cut at the cap, speech that went on continues the
                // utterance; speech after a pause as long as an ending starts one.
                self.starts_utterance = !(self.capped && start - self.last_speech < END_SILENCE);
            }
            self.last_speech = self.heard;
        }
        let Some(open) = self.open else {
            // Keep only what a clip starting at the next frame would need.
            self.drop_before(self.heard.saturating_sub(PRE_ROLL).max(self.last_cut));
            return None;
        };
        if self.heard - self.last_speech >= END_SILENCE {
            self.cut(Cut::Silence)
        } else if self.heard - open >= CAP {
            self.cut(Cut::Cap)
        } else {
            None
        }
    }

    /// Closes the open clip where the frames looked at end.
    fn cut(&mut self, cut: Cut) -> Option<Clip> {
        let start = self.open.take()?;
        let audio = self.audio[(start - self.kept) as usize..(self.heard - self.kept) as usize].to_vec();
        self.last_cut = self.heard;
        self.capped = cut == Cut::Cap;
        self.drop_before(self.heard);
        Some(Clip { start, audio, cut, starts_utterance: self.starts_utterance })
    }

    fn drop_before(&mut self, at: Ts) {
        let n = at.saturating_sub(self.kept) as usize;
        if n > 0 {
            self.audio.drain(..n);
            self.kept += n as Ts;
        }
    }
}
