//! Reads one metadata value from a GGUF model file's header, without loading
//! the model: the format is a short header, then key/value pairs, then the
//! tensors (<https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>).

use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;

use anyhow::{Context, bail};

use crate::Result;

const STRING: u32 = 8;
const ARRAY: u32 = 9;

/// The array of strings stored under `key`, or `None` when the file has no
/// such key.
pub fn string_array(path: &Path, key: &str) -> Result<Option<Vec<String>>> {
    let read = || -> Result<Option<Vec<String>>> {
        let mut r = BufReader::new(File::open(path)?);
        let (_, kvs) = start(&mut r)?;
        for _ in 0..kvs {
            let name = string(&mut r)?;
            let kind = u32(&mut r)?;
            if name != key {
                skip(&mut r, kind)?;
                continue;
            }
            let element = if kind == ARRAY { u32(&mut r)? } else { kind };
            if kind != ARRAY || element != STRING {
                bail!("`{key}` isn't an array of strings");
            }
            return (0..u64(&mut r)?).map(|_| string(&mut r)).collect::<Result<_>>().map(Some);
        }
        Ok(None)
    };
    read().with_context(|| format!("reading {key} from {}", path.display()))
}

/// Whether the header has a value under `key`.
pub fn has(path: &Path, key: &str) -> Result<bool> {
    let read = || -> Result<bool> {
        let mut r = BufReader::new(File::open(path)?);
        let (_, kvs) = start(&mut r)?;
        for _ in 0..kvs {
            let name = string(&mut r)?;
            if name == key {
                return Ok(true);
            }
            let kind = u32(&mut r)?;
            skip(&mut r, kind)?;
        }
        Ok(false)
    };
    read().with_context(|| format!("reading {} for {key}", path.display()))
}

/// Copies the model `src` to `dst` with the string `value` added under
/// `key`. The tensors are copied byte for byte; only the header grows, and
/// the data after it is padded to the file's alignment again.
pub fn copy_with_string(src: &Path, dst: &Path, key: &str, value: &str) -> Result<()> {
    let copy = || -> Result<()> {
        let mut r = Counting { inner: BufReader::new(File::open(src)?), read: 0 };
        let (tensors, kvs) = start(&mut r)?;
        let mut alignment = 32;
        for _ in 0..kvs {
            let name = string(&mut r)?;
            if name == key {
                bail!("it already has {key}");
            }
            let kind = u32(&mut r)?;
            match (name.as_str(), kind) {
                ("general.alignment", 4) => alignment = u64::from(u32(&mut r)?),
                _ => skip(&mut r, kind)?,
            }
        }
        let kvs_end = r.read;
        for _ in 0..tensors {
            string(&mut r)?;
            let dims = u32(&mut r)?;
            for _ in 0..dims {
                u64(&mut r)?;
            }
            u32(&mut r)?;
            u64(&mut r)?;
        }
        let infos_end = r.read;
        let data = infos_end.next_multiple_of(alignment);
        drop(r);

        let mut header = vec![0; infos_end as usize];
        let mut f = File::open(src)?;
        f.read_exact(&mut header)?;
        let mut out = Vec::with_capacity(header.len() + value.len() + 64);
        out.extend(&header[..16]);
        out.extend((kvs + 1).to_le_bytes());
        out.extend(&header[24..kvs_end as usize]);
        put_string(&mut out, key);
        out.extend(STRING.to_le_bytes());
        put_string(&mut out, value);
        out.extend(&header[kvs_end as usize..]);
        out.resize((out.len() as u64).next_multiple_of(alignment) as usize, 0);

        // Written beside `dst` and renamed into place, so a failed copy never
        // leaves half a model; a failed one is removed, as it is as big as the model.
        let tmp = dst.with_extension(format!("{}.part", std::process::id()));
        let written = (|| -> Result<()> {
            let mut w = io::BufWriter::new(File::create(&tmp)?);
            w.write_all(&out)?;
            io::Seek::seek(&mut f, io::SeekFrom::Start(data))?;
            io::copy(&mut f, &mut w)?;
            w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
            Ok(std::fs::rename(&tmp, dst)?)
        })();
        if written.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        written
    };
    copy().with_context(|| format!("adding {key} to {} as {}", src.display(), dst.display()))
}

fn put_string(out: &mut Vec<u8>, text: &str) {
    out.extend((text.len() as u64).to_le_bytes());
    out.extend(text.as_bytes());
}

/// Reads the magic, the version and the counts: tensors, then key/value pairs.
fn start(r: &mut impl Read) -> Result<(u64, u64)> {
    let mut magic = [0; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = u32(r)?;
    if version < 2 {
        bail!("GGUF version {version} is too old");
    }
    Ok((u64(r)?, u64(r)?))
}

/// A reader that counts the bytes read through it.
struct Counting<R> {
    inner: R,
    read: u64,
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.read += n as u64;
        Ok(n)
    }
}

fn u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn string(r: &mut impl Read) -> Result<String> {
    let n = u64(r)?;
    if n > 1 << 20 {
        bail!("a {n}-byte string in the header");
    }
    let mut b = vec![0; n as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// Reads past one value of type `kind`.
fn skip(r: &mut impl Read, kind: u32) -> Result<()> {
    let size = match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        STRING => u64(r)?,
        ARRAY => {
            let element = u32(r)?;
            for _ in 0..u64(r)? {
                skip(r, element)?;
            }
            return Ok(());
        }
        other => bail!("unknown value type {other}"),
    };
    let skipped = io::copy(&mut r.take(size), &mut io::sink())?;
    if skipped < size {
        bail!("the header ends early");
    }
    Ok(())
}

/// Writes a GGUF header holding only `entries` under `key`, for tests.
#[cfg(test)]
pub(crate) fn write_string_array(path: &Path, key: &str, entries: &[&str]) {
    let s = |out: &mut Vec<u8>, text: &str| {
        out.extend((text.len() as u64).to_le_bytes());
        out.extend(text.as_bytes());
    };
    let mut h = b"GGUF".to_vec();
    h.extend(3u32.to_le_bytes());
    h.extend(0u64.to_le_bytes());
    h.extend(1u64.to_le_bytes());
    s(&mut h, key);
    h.extend(ARRAY.to_le_bytes());
    h.extend(STRING.to_le_bytes());
    h.extend((entries.len() as u64).to_le_bytes());
    for e in entries {
        s(&mut h, e);
    }
    std::fs::write(path, h).unwrap();
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    fn s(out: &mut Vec<u8>, text: &str) {
        out.extend((text.len() as u64).to_le_bytes());
        out.extend(text.as_bytes());
    }

    /// A header with a u32, an array of u16, an array of strings under `key`,
    /// and a string.
    pub(in crate::nemo) fn header(key: &str) -> Vec<u8> {
        let mut h = b"GGUF".to_vec();
        h.extend(3u32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        h.extend(4u64.to_le_bytes());
        s(&mut h, "general.alignment");
        h.extend(4u32.to_le_bytes());
        h.extend(32u32.to_le_bytes());
        s(&mut h, "numbers");
        h.extend(ARRAY.to_le_bytes());
        h.extend(2u32.to_le_bytes());
        h.extend(3u64.to_le_bytes());
        h.extend([1, 0, 2, 0, 3, 0]);
        s(&mut h, key);
        h.extend(ARRAY.to_le_bytes());
        h.extend(STRING.to_le_bytes());
        h.extend(2u64.to_le_bytes());
        s(&mut h, "en-US:0");
        s(&mut h, "zh-CN:4");
        s(&mut h, "general.name");
        h.extend(STRING.to_le_bytes());
        s(&mut h, "x");
        h
    }

    #[test]
    fn finds_a_string_array_past_other_values() {
        let dir = tempfile_dir("find");
        let path = dir.join("m.gguf");
        std::fs::write(&path, header("asr.rnnt.prompt_dictionary")).unwrap();
        let got = string_array(&path, "asr.rnnt.prompt_dictionary").unwrap();
        assert_eq!(got.unwrap(), ["en-US:0", "zh-CN:4"]);
        assert_eq!(string_array(&path, "missing").unwrap(), None);
        let err = format!("{:#}", string_array(&path, "general.alignment").unwrap_err());
        assert!(err.contains("isn't an array of strings"), "{err}");

        std::fs::write(&path, b"GGML....").unwrap();
        assert!(format!("{:#}", string_array(&path, "x").unwrap_err()).contains("not a GGUF file"));
        let mut cut = header("k");
        cut.truncate(60);
        std::fs::write(&path, cut).unwrap();
        assert!(string_array(&path, "k").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_string_is_added_and_the_tensors_stay_aligned() {
        let dir = tempfile_dir("add");
        let (src, dst) = (dir.join("m.gguf"), dir.join("m.boost.gguf"));
        // The header above, then one tensor of 16 bytes, padded to 32.
        let mut h = header("k");
        h[8..16].copy_from_slice(&1u64.to_le_bytes());
        s(&mut h, "t");
        h.extend(1u32.to_le_bytes());
        h.extend(16u64.to_le_bytes());
        h.extend(0u32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        h.resize(h.len().next_multiple_of(32), 0);
        let data: Vec<u8> = (0..16).collect();
        h.extend(&data);
        std::fs::write(&src, &h).unwrap();

        assert!(!has(&src, "asr.tokenizer.spm_model").unwrap());
        copy_with_string(&src, &dst, "asr.tokenizer.spm_model", "c3Bt").unwrap();
        let got = std::fs::read(&dst).unwrap();
        assert!(has(&dst, "asr.tokenizer.spm_model").unwrap());
        assert_eq!(string_array(&dst, "k").unwrap().unwrap(), ["en-US:0", "zh-CN:4"], "the rest is kept");
        assert_eq!(u64::from_le_bytes(got[16..24].try_into().unwrap()), 5);
        assert!(got.ends_with(&data));
        assert_eq!((got.len() - data.len()) % 32, 0, "the data starts aligned");
        let err = format!("{:#}", copy_with_string(&dst, &src, "asr.tokenizer.spm_model", "x").unwrap_err());
        assert!(err.contains("already has"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A folder of the test's own: tests run side by side.
    fn tempfile_dir(test: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("pam-gguf-{}-{test}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
