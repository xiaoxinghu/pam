//! The meeting clock. All times are 16 kHz sample indices since the meeting
//! started ([`Ts`]); seconds and `mm:ss` exist only for display and files.

pub use pam_speech::{SAMPLE_RATE, Ts};

/// Samples in one second.
pub const SECOND: Ts = SAMPLE_RATE as Ts;

/// `mm:ss`, or `h:mm:ss` from the first hour on.
pub fn mmss(ts: Ts) -> String {
    let s = ts / SECOND;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{:02}:{:02}", s / 60, s % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_mmss() {
        assert_eq!(mmss(0), "00:00");
        assert_eq!(mmss(61 * SECOND + 15_999), "01:01");
        assert_eq!(mmss(3_725 * SECOND), "1:02:05");
    }
}
