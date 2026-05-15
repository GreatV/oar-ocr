//! Auto-download of OCR model files from ModelScope (feature `auto-download`).
//!
//! When this feature is enabled, callers can pass *just a filename* (such as
//! `"pp-ocrv5_mobile_det.onnx"`) anywhere the OCR builders accept a model
//! path. If the file is not already on disk it is fetched from
//! [`greatv/oar-ocr`](https://www.modelscope.cn/models/greatv/oar-ocr),
//! its SHA-256 verified against [`registry::REGISTRY`], and cached under
//! `$OAR_HOME` (default `~/.oar`).
//!
//! ## Lookup rules
//!
//! [`resolve_path`] is the single entry point used by the builders:
//!
//! 1. If the input path exists on disk and refers to a file, it is returned
//!    as-is (no hash check) — users keep full control over local files.
//! 2. Otherwise, if the file *name* matches an entry in the registry, the
//!    cached copy under `$OAR_HOME` is reused (verified) or downloaded and
//!    verified.
//! 3. Otherwise, the path is returned unchanged so the caller produces its
//!    usual "model not found" error.
//!
//! ## Cache location
//!
//! - `$OAR_HOME` environment variable (if set), else
//! - `<home>/.oar`
//!
//! The directory is created lazily on first download.

use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::core::errors::OCRError;

pub mod registry;

pub use registry::{DEFAULT_REVISION, MODELSCOPE_REPO, REGISTRY};

/// A registered file mirrored to ModelScope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// File name as stored in the repo and in the cache directory.
    pub name: &'static str,
    /// Lowercase hex-encoded SHA-256 of the file contents.
    pub sha256: &'static str,
    /// Size in bytes.
    pub size: u64,
}

/// Environment variable used to override the cache directory.
pub const OAR_HOME_ENV: &str = "OAR_HOME";

/// Default cache directory name placed under the user's home dir.
const DEFAULT_CACHE_SUBDIR: &str = ".oar";

const DOWNLOAD_RETRIES: u32 = 3;
const READ_BUFFER_BYTES: usize = 64 * 1024;
/// Whole-request timeout for a single download attempt. Sized to allow the
/// largest registered file (~1.8 GB) to complete on slow links (~1 MB/s);
/// the retry loop applies this per attempt.
const REQUEST_TIMEOUT_SECS: u64 = 30 * 60;
const CONNECT_TIMEOUT_SECS: u64 = 30;

/// Look up an entry by file name. Returns `None` if the file isn't registered.
pub fn find(name: &str) -> Option<&'static Entry> {
    REGISTRY
        .binary_search_by_key(&name, |e| e.name)
        .ok()
        .map(|idx| &REGISTRY[idx])
}

/// Returns the cache directory used to store auto-downloaded models.
///
/// Resolution order:
/// 1. `$OAR_HOME`, if set and non-empty.
/// 2. `<home>/.oar` where `<home>` is the platform user home directory.
///
/// Falls back to `./.oar` (relative to the current working directory) when
/// the user home cannot be determined — this matches what the repository's
/// existing examples already use.
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = env::var_os(OAR_HOME_ENV) {
        let dir = PathBuf::from(dir);
        if !dir.as_os_str().is_empty() {
            return dir;
        }
    }
    dirs::home_dir()
        .map(|h| h.join(DEFAULT_CACHE_SUBDIR))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_SUBDIR))
}

/// Resolve a user-supplied path through the auto-download cache.
///
/// See module docs for the lookup rules. Returns the resolved path (either
/// the original `input`, an existing cache entry, or a freshly downloaded
/// file).
pub fn resolve_path(input: impl AsRef<Path>) -> Result<PathBuf, OCRError> {
    let input = input.as_ref();

    // Rule 1: trust paths that already exist on disk.
    if input.is_file() {
        return Ok(input.to_path_buf());
    }

    // Rule 2: registry lookup keyed on the file name component.
    let name = match input.file_name().and_then(|s| s.to_str()) {
        Some(n) => n,
        None => return Ok(input.to_path_buf()),
    };

    // Only match when the user gave just a filename (no parent component) or
    // when the parent is the configured cache directory. This avoids quietly
    // overriding a user's explicit path like `./models/pp-ocrv4_mobile_det.onnx`.
    let parent_is_bare = input.parent().is_none_or(|p| p.as_os_str().is_empty());
    let cache = cache_dir();
    let parent_is_cache = input.parent() == Some(cache.as_path());

    if !(parent_is_bare || parent_is_cache) {
        return Ok(input.to_path_buf());
    }

    if let Some(entry) = find(name) {
        return fetch_entry(entry);
    }

    Ok(input.to_path_buf())
}

/// Fetch a registered file by name, returning its path in the cache.
///
/// If the file is already present and its SHA-256 matches, no network access
/// occurs; otherwise it is downloaded from ModelScope and verified.
pub fn fetch(name: &str) -> Result<PathBuf, OCRError> {
    let entry = find(name).ok_or_else(|| OCRError::ConfigError {
        message: format!(
            "model file `{}` is not registered for auto-download. \
             Pass an explicit path or add an entry to oar_ocr_core::core::download::registry::REGISTRY.",
            name
        ),
    })?;
    fetch_entry(entry)
}

fn fetch_entry(entry: &Entry) -> Result<PathBuf, OCRError> {
    let dir = cache_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        return Err(OCRError::Io(io::Error::new(
            e.kind(),
            format!(
                "failed to create OAR cache directory `{}`: {}",
                dir.display(),
                e
            ),
        )));
    }
    let target = dir.join(entry.name);

    if cached_file_matches(&target, entry)? {
        return Ok(target);
    }

    download_and_verify(entry, &target)?;
    Ok(target)
}

fn cached_file_matches(path: &Path, entry: &Entry) -> Result<bool, OCRError> {
    let meta = match fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(io_with_context(e, format!("stat `{}`", path.display()))),
    };
    if !meta.is_file() {
        return Ok(false);
    }
    if meta.len() != entry.size {
        tracing::warn!(
            path = %path.display(),
            expected_size = entry.size,
            actual_size = meta.len(),
            "cached model has wrong size; redownloading"
        );
        return Ok(false);
    }

    // Fast path: if a sidecar from a previous verified download/check exists
    // and records the expected hash, skip rehashing the (potentially 1.8 GB)
    // file. The sidecar is written under the same cache directory we control,
    // so the threat model matches "trust the cache once we've vouched for it".
    if sidecar_records_hash(path, entry.sha256) {
        return Ok(true);
    }

    match sha256_file(path) {
        Ok(hash) if hash == entry.sha256 => {
            // Remember the verification so future loads skip the rehash.
            if let Err(e) = write_sidecar(path, entry.sha256) {
                tracing::debug!(
                    path = %path.display(),
                    error = %e,
                    "failed to write sha256 sidecar; cache will rehash next time"
                );
            }
            Ok(true)
        }
        Ok(hash) => {
            tracing::warn!(
                path = %path.display(),
                expected = entry.sha256,
                actual = %hash,
                "cached model sha256 mismatch; redownloading"
            );
            Ok(false)
        }
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "failed to hash cached model; redownloading"
            );
            Ok(false)
        }
    }
}

fn sidecar_path(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    Some(path.with_file_name(format!(".{name}.sha256")))
}

fn sidecar_records_hash(path: &Path, expected: &str) -> bool {
    let Some(sidecar) = sidecar_path(path) else {
        return false;
    };
    match fs::read_to_string(&sidecar) {
        Ok(contents) => contents.trim().eq_ignore_ascii_case(expected),
        Err(_) => false,
    }
}

fn write_sidecar(path: &Path, hash: &str) -> io::Result<()> {
    let sidecar = sidecar_path(path)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no filename for sidecar"))?;
    fs::write(&sidecar, hash)
}

fn download_and_verify(entry: &Entry, target: &Path) -> Result<(), OCRError> {
    let url = format!(
        "https://www.modelscope.cn/api/v1/models/{}/repo?Revision={}&FilePath={}",
        MODELSCOPE_REPO, DEFAULT_REVISION, entry.name,
    );

    // Build the agent once so connection pooling and HTTPS handshakes survive
    // across retry attempts.
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(REQUEST_TIMEOUT_SECS)))
        .timeout_connect(Some(Duration::from_secs(CONNECT_TIMEOUT_SECS)))
        .build()
        .new_agent();

    let mut last_err: Option<OCRError> = None;
    for attempt in 1..=DOWNLOAD_RETRIES {
        tracing::info!(
            file = entry.name,
            size = entry.size,
            attempt,
            "downloading from ModelScope"
        );
        match download_attempt(&agent, &url, entry, target) {
            Ok(()) => return Ok(()),
            Err(e) => {
                tracing::warn!(
                    file = entry.name,
                    attempt,
                    error = %e,
                    "download attempt failed",
                );
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| OCRError::ConfigError {
        message: format!("download of `{}` failed after retries", entry.name),
    }))
}

/// Monotonic counter used to keep concurrent in-process downloads of the same
/// entry from sharing a temp path. Combined with the PID it gives a unique
/// suffix without pulling in a `rand` dependency.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_tmp_path(target: &Path, entry: &Entry) -> PathBuf {
    let counter = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    target.with_file_name(format!(
        ".{}.{}.{}.part",
        entry.name,
        std::process::id(),
        counter
    ))
}

/// RAII guard that deletes a temp file on drop unless explicitly disarmed.
/// Keeps `$OAR_HOME` from accumulating stale `.part` files when a download
/// fails mid-stream (read error, write error, hash mismatch, …).
struct TempFileGuard {
    path: Option<PathBuf>,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn path(&self) -> &Path {
        // Only `disarm` clears `path`, and it consumes `self`, so anywhere
        // we still hold the guard the path is present.
        self.path.as_deref().expect("guard already disarmed")
    }

    /// Hand off ownership of the temp file to a successful rename; nothing
    /// gets deleted on drop after this point.
    fn disarm(mut self) {
        self.path = None;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_file(path);
        }
    }
}

fn download_attempt(
    agent: &ureq::Agent,
    url: &str,
    entry: &Entry,
    target: &Path,
) -> Result<(), OCRError> {
    let response = agent
        .get(url)
        .call()
        .map_err(|e| network_error(format!("GET {}", url), e))?;

    let mut body = response.into_body().into_reader();

    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let tmp = unique_tmp_path(target, entry);
    let mut file = File::create(&tmp).map_err(|e| {
        io_with_context(
            e,
            format!(
                "create temp download `{}` in `{}`",
                tmp.display(),
                parent.display()
            ),
        )
    })?;
    // From here on, any early return must not leak the temp file.
    let guard = TempFileGuard::new(tmp);

    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    let mut buf = vec![0u8; READ_BUFFER_BYTES];
    let mut written: u64 = 0;
    loop {
        let n = body
            .read(&mut buf)
            .map_err(|e| io_with_context(e, format!("read body for `{}`", entry.name)))?;
        if n == 0 {
            break;
        }
        sha2::Digest::update(&mut hasher, &buf[..n]);
        file.write_all(&buf[..n])
            .map_err(|e| io_with_context(e, format!("write `{}`", guard.path().display())))?;
        written += n as u64;
    }
    file.sync_all()
        .map_err(|e| io_with_context(e, format!("sync `{}`", guard.path().display())))?;
    drop(file);

    if written != entry.size {
        return Err(OCRError::ConfigError {
            message: format!(
                "downloaded `{}` is {} bytes but the registry expects {}",
                entry.name, written, entry.size
            ),
        });
    }

    let actual_hash = encode_hex(&sha2::Digest::finalize(hasher));
    if actual_hash != entry.sha256 {
        return Err(OCRError::ConfigError {
            message: format!(
                "sha256 mismatch for `{}`: expected {}, got {}",
                entry.name, entry.sha256, actual_hash
            ),
        });
    }

    fs::rename(guard.path(), target).map_err(|e| {
        io_with_context(
            e,
            format!(
                "move `{}` -> `{}`",
                guard.path().display(),
                target.display()
            ),
        )
    })?;
    // The temp path no longer exists (renamed onto `target`); disarm so
    // Drop doesn't try to remove a now-missing file.
    guard.disarm();

    // Record the verified hash next to the file so subsequent loads can skip
    // the expensive rehash. Best-effort: a failure here only costs a rehash.
    if let Err(e) = write_sidecar(target, entry.sha256) {
        tracing::debug!(
            path = %target.display(),
            error = %e,
            "failed to write sha256 sidecar after download"
        );
    }
    Ok(())
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    let mut buf = vec![0u8; READ_BUFFER_BYTES];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        sha2::Digest::update(&mut hasher, &buf[..n]);
    }
    Ok(encode_hex(&sha2::Digest::finalize(hasher)))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

fn io_with_context(e: io::Error, ctx: String) -> OCRError {
    OCRError::Io(io::Error::new(e.kind(), format!("{ctx}: {e}")))
}

fn network_error(ctx: String, err: ureq::Error) -> OCRError {
    OCRError::Io(io::Error::other(format!("{ctx}: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_known_entry() {
        let entry = find("ppocrv5_en_dict.txt").expect("registered");
        assert_eq!(entry.size, 1416);
    }

    #[test]
    fn find_unknown_entry_returns_none() {
        assert!(find("does-not-exist.onnx").is_none());
    }

    #[test]
    fn resolve_existing_file_returns_input() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("local.onnx");
        std::fs::write(&f, b"hi").unwrap();
        let resolved = resolve_path(&f).unwrap();
        assert_eq!(resolved, f);
    }

    #[test]
    fn resolve_explicit_path_passthrough_for_unknown() {
        // A nested path that doesn't exist and isn't registered must be
        // returned verbatim so the caller's normal error fires.
        let p = PathBuf::from("/nonexistent/dir/some_random_model.onnx");
        let resolved = resolve_path(&p).unwrap();
        assert_eq!(resolved, p);
    }

    #[test]
    fn resolve_bare_name_unknown_does_not_consult_network() {
        // No registry hit, no existing file → returned as-is.
        let p = PathBuf::from("not-in-registry.onnx");
        let resolved = resolve_path(&p).unwrap();
        assert_eq!(resolved, p);
    }

    #[test]
    fn cache_dir_honours_env_override() {
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var(OAR_HOME_ENV, dir.path());
        }
        assert_eq!(cache_dir(), dir.path());
        unsafe {
            std::env::remove_var(OAR_HOME_ENV);
        }
    }

    #[test]
    fn cached_file_matches_detects_size_and_hash_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dummy.bin");

        // Fake registry entry pinned to known SHA-256 of "hello\n" (6 bytes).
        let entry = Entry {
            name: "dummy.bin",
            // sha256 of b"hello\n"
            sha256: "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
            size: 6,
        };

        // Missing file → no match (no network needed).
        assert!(!cached_file_matches(&path, &entry).unwrap());

        // Correct contents → match.
        std::fs::write(&path, b"hello\n").unwrap();
        assert!(cached_file_matches(&path, &entry).unwrap());

        // Same hash but wrong size when we lie about expected size → no match.
        let mismatched_size = Entry { size: 99, ..entry };
        assert!(!cached_file_matches(&path, &mismatched_size).unwrap());

        // Wrong contents → no match (hash differs). Clear any sidecar left
        // behind by the earlier matches so the rehash actually runs.
        let sidecar = sidecar_path(&path).unwrap();
        let _ = std::fs::remove_file(&sidecar);
        std::fs::write(&path, b"world!").unwrap();
        let same_len = Entry { size: 6, ..entry };
        assert!(!cached_file_matches(&path, &same_len).unwrap());
    }

    #[test]
    fn cached_file_matches_writes_sidecar_and_uses_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dummy.bin");
        let entry = Entry {
            name: "dummy.bin",
            sha256: "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
            size: 6,
        };
        std::fs::write(&path, b"hello\n").unwrap();

        // First match triggers a real hash + writes the sidecar.
        assert!(cached_file_matches(&path, &entry).unwrap());
        let sidecar = sidecar_path(&path).unwrap();
        assert_eq!(std::fs::read_to_string(&sidecar).unwrap(), entry.sha256);

        // Tampering with the file but keeping size: sidecar lets us trust the
        // (now stale) cache. This is the deliberate tradeoff — see module docs.
        std::fs::write(&path, b"world!").unwrap();
        assert!(cached_file_matches(&path, &entry).unwrap());

        // If the sidecar disagrees with the expected hash, we fall back to
        // rehashing and (here) reject the cache.
        std::fs::write(&sidecar, "deadbeef").unwrap();
        assert!(!cached_file_matches(&path, &entry).unwrap());
    }

    #[test]
    fn temp_file_guard_removes_on_drop_and_keeps_when_disarmed() {
        let dir = tempfile::tempdir().unwrap();

        // Drop-without-disarm deletes the file.
        let p = dir.path().join("dropme.part");
        std::fs::write(&p, b"x").unwrap();
        drop(TempFileGuard::new(p.clone()));
        assert!(!p.exists(), "guard should remove the temp file on drop");

        // Disarm keeps the file.
        let p = dir.path().join("keepme.part");
        std::fs::write(&p, b"x").unwrap();
        let guard = TempFileGuard::new(p.clone());
        guard.disarm();
        assert!(p.exists(), "disarmed guard must not delete the file");

        // Missing temp path on drop is silently ignored (no panic).
        let p = dir.path().join("ghost.part");
        drop(TempFileGuard::new(p));
    }

    #[test]
    fn unique_tmp_path_never_repeats_in_process() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model.onnx");
        let entry = Entry {
            name: "model.onnx",
            sha256: "00",
            size: 0,
        };
        let a = unique_tmp_path(&target, &entry);
        let b = unique_tmp_path(&target, &entry);
        assert_ne!(a, b);
        let pid = std::process::id().to_string();
        assert!(a.to_string_lossy().contains(&pid));
    }

    #[test]
    fn fetch_unregistered_name_returns_config_error() {
        let err = fetch("does-not-exist.onnx").unwrap_err();
        match err {
            OCRError::ConfigError { message } => {
                assert!(message.contains("not registered"));
            }
            other => panic!("expected ConfigError, got {other:?}"),
        }
    }
}
