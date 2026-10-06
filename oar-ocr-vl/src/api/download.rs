//! Snapshot download of VL checkpoints (feature `auto-download`).
//!
//! [`AnyPageParser::from_pretrained`](crate::AnyPageParser::from_pretrained)
//! downloads a checkpoint repo by its [`AnyPageParserModel`](crate::AnyPageParserModel)
//! ID when it is not cached, then loads it through the regular directory
//! path. Snapshots land under `$OAR_HOME/models/<org>/<name>` (`$OAR_HOME`
//! defaults to `~/.oar`, as in oar-ocr-core), each file is verified against
//! the hash the source API provides (ModelScope always publishes SHA-256;
//! Hugging Face publishes it for LFS files), and complete cached files are
//! never re-downloaded.
//!
//! Repos are pinned to a revision — `master` on ModelScope and `main` on
//! Hugging Face by default, overridable per call — and the revision used is
//! recorded in a `.oar-revision` marker inside the snapshot directory.

use crate::api::error::Error;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Default layout checkpoint downloaded for layout-composed models.
pub const DEFAULT_LAYOUT_REPO: &str = "PaddlePaddle/PP-DocLayoutV3_safetensors";

/// Checkpoints ModelScope publishes under a different organization than
/// Hugging Face. The mirrors were verified to carry the same files, with
/// matching safetensors hashes; only `zai-org/GLM-OCR` has one.
const MODELSCOPE_ALIASES: &[(&str, &str)] = &[("zai-org/GLM-OCR", "ZhipuAI/GLM-OCR")];

/// Checkpoints ModelScope does not carry at all; selecting ModelScope
/// downloads these from Hugging Face instead. (`Tencent-Hunyuan/HunyuanOCR`
/// exists on ModelScope but is a different, 1.0-style repo layout rather than
/// a mirror of the Hugging Face one, so it is deliberately not aliased.)
const HF_FALLBACK_REPOS: &[&str] = &[
    "tencent/HunyuanOCR",
    "tencent/WeVisDoc-2B",
    "tencent/WeVisDoc-4B",
];

/// Where checkpoints are downloaded from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DownloadSource {
    /// ModelScope (`www.modelscope.cn`), the default.
    #[default]
    ModelScope,
    /// Hugging Face (`huggingface.co`).
    HuggingFace,
}

impl DownloadSource {
    /// The revision downloaded when none is pinned in the options.
    fn default_revision(self) -> &'static str {
        match self {
            Self::ModelScope => "master",
            Self::HuggingFace => "main",
        }
    }

    fn files_url(self, repo: &str, revision: &str) -> String {
        match self {
            Self::ModelScope => format!(
                "https://www.modelscope.cn/api/v1/models/{}/repo/files?Revision={}&Recursive=true",
                encode_path(repo),
                encode_component(revision)
            ),
            Self::HuggingFace => format!(
                "https://huggingface.co/api/models/{}/tree/{}?recursive=true",
                encode_path(repo),
                encode_path(revision)
            ),
        }
    }

    fn file_url(self, repo: &str, revision: &str, path: &str) -> String {
        match self {
            Self::ModelScope => format!(
                "https://www.modelscope.cn/api/v1/models/{}/repo?Revision={}&FilePath={}",
                encode_path(repo),
                encode_component(revision),
                encode_path(path)
            ),
            Self::HuggingFace => format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                encode_path(repo),
                encode_path(revision),
                encode_path(path)
            ),
        }
    }
}

/// Percent-encodes one URL component; only unreserved characters pass
/// through. Slashes are the caller's business — see [`encode_path`].
fn encode_component(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push(HEX[(byte >> 4) as usize] as char);
                out.push(HEX[(byte & 0xf) as usize] as char);
            }
        }
    }
    out
}

/// Encodes a multi-segment path (repo id, revision like `refs/pr/123`, or a
/// file path), keeping the `/` separators literal.
fn encode_path(value: &str) -> String {
    value
        .split('/')
        .map(encode_component)
        .collect::<Vec<_>>()
        .join("/")
}

/// Download options for [`AnyPageParser::from_pretrained`](crate::AnyPageParser::from_pretrained).
///
/// Defaults download from ModelScope at the source's default revision, and
/// the layout-composed models get [`DEFAULT_LAYOUT_REPO`]. Every knob is
/// optional and `None` keeps its default.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct AnyPageParserPretrainedOptions {
    /// Checkpoint source. `None` means ModelScope.
    pub source: Option<DownloadSource>,
    /// Revision to pin. `None` means the source's default (`master` on
    /// ModelScope, `main` on Hugging Face).
    pub revision: Option<String>,
    /// Layout checkpoint repo downloaded for layout-composed models. `None`
    /// means [`DEFAULT_LAYOUT_REPO`].
    pub layout: Option<String>,
    /// Use a local PP-DocLayout directory instead of downloading one.
    pub layout_dir: Option<PathBuf>,
}

impl AnyPageParserPretrainedOptions {
    /// Download from a specific source.
    pub fn with_source(mut self, source: DownloadSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Pin a specific revision for the model repos; the layout checkpoint
    /// always downloads at its source's default revision.
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Download a different layout checkpoint repo for layout-composed models.
    pub fn with_layout(mut self, repo: impl Into<String>) -> Self {
        self.layout = Some(repo.into());
        self
    }

    /// Use a local layout directory instead of downloading one.
    pub fn with_layout_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.layout_dir = Some(dir.into());
        self
    }

    pub(crate) fn source(&self) -> DownloadSource {
        self.source.unwrap_or_default()
    }

    pub(crate) fn layout(&self) -> &str {
        self.layout.as_deref().unwrap_or(DEFAULT_LAYOUT_REPO)
    }
}

/// One file in a repo snapshot.
#[derive(Debug, PartialEq, Eq)]
struct SnapshotFile {
    path: String,
    size: u64,
    /// SHA-256 of the contents when the source API publishes it.
    sha256: Option<String>,
}

#[derive(Deserialize)]
struct ModelScopeListing {
    #[serde(rename = "Data")]
    data: ModelScopeData,
}

#[derive(Deserialize)]
struct ModelScopeData {
    #[serde(rename = "Files")]
    files: Vec<ModelScopeEntry>,
}

#[derive(Deserialize)]
struct ModelScopeEntry {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Sha256")]
    sha256: Option<String>,
    #[serde(rename = "Size")]
    size: u64,
    #[serde(rename = "Type", default)]
    kind: String,
}

#[derive(Deserialize)]
struct HuggingFaceEntry {
    path: String,
    size: Option<u64>,
    lfs: Option<HuggingFaceLfs>,
    #[serde(rename = "type", default)]
    kind: String,
}

#[derive(Deserialize)]
struct HuggingFaceLfs {
    oid: Option<String>,
}

/// Parses a ModelScope `repo/files` listing into snapshot files.
fn parse_modelscope_listing(body: &str) -> Result<Vec<SnapshotFile>, Error> {
    let listing: ModelScopeListing = serde_json::from_str(body)
        .map_err(|error| Error::config(format!("parse ModelScope file listing: {error}")))?;
    Ok(listing
        .data
        .files
        .into_iter()
        .filter(|entry| entry.kind == "blob")
        .map(|entry| SnapshotFile {
            path: entry.path,
            size: entry.size,
            sha256: entry.sha256,
        })
        .collect())
}

/// Parses a Hugging Face `tree` listing into snapshot files. LFS entries
/// carry a SHA-256 in `lfs.oid`; plain-git entries only report a size.
fn parse_huggingface_listing(body: &str) -> Result<Vec<SnapshotFile>, Error> {
    let entries: Vec<HuggingFaceEntry> = serde_json::from_str(body)
        .map_err(|error| Error::config(format!("parse Hugging Face file listing: {error}")))?;
    Ok(entries
        .into_iter()
        .filter(|entry| entry.kind == "file")
        .map(|entry| SnapshotFile {
            path: entry.path,
            size: entry.size.unwrap_or(0),
            sha256: entry.lfs.and_then(|lfs| lfs.oid),
        })
        .collect())
}

/// Cache root shared with oar-ocr-core: `$OAR_HOME`, else `~/.oar`.
fn cache_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("OAR_HOME") {
        let dir = PathBuf::from(dir);
        if !dir.as_os_str().is_empty() {
            return dir;
        }
    }
    dirs::home_dir()
        .map(|home| home.join(".oar"))
        .unwrap_or_else(|| PathBuf::from(".oar"))
}

/// Maps a repo ID to its snapshot directory under the cache root.
fn snapshot_dir(root: &Path, repo: &str) -> Result<PathBuf, Error> {
    let (org, name) = repo.split_once('/').ok_or_else(|| {
        Error::config(format!(
            "model id {repo:?} is not an <org>/<name> repository id"
        ))
    })?;
    Ok(root.join("models").join(org).join(name))
}

const DOWNLOAD_RETRIES: u32 = 3;
const READ_BUFFER_BYTES: usize = 64 * 1024;
const REQUEST_TIMEOUT_SECS: u64 = 30 * 60;
const CONNECT_TIMEOUT_SECS: u64 = 30;

/// Resolves which source and remote repo a download actually uses: ModelScope
/// aliases point at their verified mirror, and repos ModelScope lacks fall
/// back to Hugging Face with a log line.
fn resolve_remote(source: DownloadSource, repo: &str) -> (DownloadSource, String) {
    if source == DownloadSource::HuggingFace {
        return (source, repo.to_string());
    }
    if let Some((_, alias)) = MODELSCOPE_ALIASES.iter().find(|(id, _)| *id == repo) {
        return (source, alias.to_string());
    }
    if HF_FALLBACK_REPOS.contains(&repo) {
        tracing::info!("{repo} is not published on ModelScope; downloading from Hugging Face");
        return (DownloadSource::HuggingFace, repo.to_string());
    }
    (source, repo.to_string())
}

/// Downloads (or reuses) the pinned snapshot of `repo` and returns its
/// directory in the cache.
pub(crate) fn snapshot(
    source: DownloadSource,
    repo: &str,
    revision: Option<&str>,
) -> Result<PathBuf, Error> {
    let (source, remote) = resolve_remote(source, repo);
    let revision = revision.unwrap_or_else(|| source.default_revision());
    // The cache stays keyed by the model id, so the source chosen never
    // changes where a snapshot lives.
    let dir = snapshot_dir(&cache_root(), repo)?;
    fs::create_dir_all(&dir).map_err(|error| {
        Error::Io(io::Error::new(
            error.kind(),
            format!("create snapshot directory `{}`: {}", dir.display(), error),
        ))
    })?;

    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(REQUEST_TIMEOUT_SECS)))
        .timeout_connect(Some(Duration::from_secs(CONNECT_TIMEOUT_SECS)))
        .build()
        .new_agent();

    let files = list_files(&agent, source, &remote, revision)?;
    prune_stale_files(&dir, &files)?;
    for file in &files {
        ensure_file(&agent, source, &remote, revision, &dir, file)?;
    }

    let marker = dir.join(".oar-revision");
    let current = format!("{remote}\n{revision}\n");
    if fs::read_to_string(&marker).unwrap_or_default() != current {
        fs::write(&marker, &current).map_err(|error| {
            Error::Io(io::Error::new(
                error.kind(),
                format!("write `{}`: {}", marker.display(), error),
            ))
        })?;
    }
    Ok(dir)
}

/// Removes cached files the new listing no longer carries, so a snapshot
/// always matches the requested revision. Our own sidecar and revision-marker
/// files stay; directories left empty are removed.
fn prune_stale_files(dir: &Path, files: &[SnapshotFile]) -> Result<(), Error> {
    let listed: HashSet<&str> = files.iter().map(|file| file.path.as_str()).collect();
    prune_entry(dir, dir, &listed).map(|_| ()).map_err(|error| {
        Error::Io(io::Error::new(
            error.kind(),
            format!("prune `{}`: {}", dir.display(), error),
        ))
    })
}

/// Returns whether the entry was removed; directories report `false` so their
/// parent never disappears from under a surviving sibling.
fn prune_entry(root: &Path, entry_path: &Path, listed: &HashSet<&str>) -> io::Result<bool> {
    let metadata = fs::symlink_metadata(entry_path)?;
    if !metadata.is_dir() {
        let Some(relative) = entry_path.strip_prefix(root).ok().and_then(|p| p.to_str()) else {
            return Ok(false);
        };
        if listed.contains(relative) || is_ours(relative) {
            return Ok(false);
        }
        tracing::info!(path = %entry_path.display(), "pruned file no longer in the requested revision");
        fs::remove_file(entry_path)?;
        return Ok(true);
    }
    let mut removed_any = false;
    for child in fs::read_dir(entry_path)? {
        let child = child?;
        removed_any |= prune_entry(root, &child.path(), listed)?;
    }
    if removed_any && entry_path != root && fs::read_dir(entry_path)?.next().is_none() {
        // Best effort: drop directories the prune emptied out.
        let _ = fs::remove_dir(entry_path);
    }
    Ok(false)
}

/// Whether a relative path is one of this module's bookkeeping files.
fn is_ours(relative: &str) -> bool {
    let name = relative.rsplit('/').next().unwrap_or(relative);
    name == ".oar-revision" || (name.starts_with('.') && name.ends_with(".sha256"))
}

/// Lists a repo's files through the source API.
fn list_files(
    agent: &ureq::Agent,
    source: DownloadSource,
    repo: &str,
    revision: &str,
) -> Result<Vec<SnapshotFile>, Error> {
    let mut url = source.files_url(repo, revision);
    let mut files = Vec::new();
    loop {
        let response = agent
            .get(&url)
            .call()
            .map_err(|error| listing_error(source, repo, error))?;
        // The tree endpoints paginate through a `Link: <...>; rel="next"`
        // header; keep fetching until there is no next page.
        let next = response
            .headers()
            .get("link")
            .and_then(|value| value.to_str().ok())
            .and_then(next_page)
            .map(str::to_string);
        let body = response
            .into_body()
            .read_to_string()
            .map_err(|error| Error::Io(io::Error::other(format!("read {url}: {error}"))))?;
        files.extend(match source {
            DownloadSource::ModelScope => parse_modelscope_listing(&body)?,
            DownloadSource::HuggingFace => parse_huggingface_listing(&body)?,
        });
        match next {
            Some(next_url) => url = next_url,
            None => return Ok(files),
        }
    }
}

/// Extracts the `rel="next"` URL from a `Link` header, when present.
fn next_page(link: &str) -> Option<&str> {
    link.split(',').find_map(|part| {
        let (url, rel) = part.split_once(';')?;
        rel.trim()
            .eq_ignore_ascii_case("rel=\"next\"")
            .then(|| url.trim().trim_start_matches('<').trim_end_matches('>'))
    })
}

/// Names a repo the source does not carry, and points at the other source.
fn listing_error(source: DownloadSource, repo: &str, error: ureq::Error) -> Error {
    let base = format!("list {repo} files on {source:?}: {error}");
    if matches!(&error, ureq::Error::StatusCode(404)) {
        return Error::config(format!(
            "{base}; the repository is not published under this id on the selected source — \
             try DownloadSource::HuggingFace or DownloadSource::ModelScope"
        ));
    }
    Error::Io(io::Error::other(base))
}

/// Ensures one snapshot file is present, verified, and untouched since.
fn ensure_file(
    agent: &ureq::Agent,
    source: DownloadSource,
    repo: &str,
    revision: &str,
    dir: &Path,
    file: &SnapshotFile,
) -> Result<(), Error> {
    let target = dir.join(&file.path);
    if cached_file_matches(&target, file)? {
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            Error::Io(io::Error::new(
                error.kind(),
                format!("create `{}`: {}", parent.display(), error),
            ))
        })?;
    }

    let url = source.file_url(repo, revision, &file.path);
    let mut last_error: Option<Error> = None;
    for attempt in 1..=DOWNLOAD_RETRIES {
        tracing::info!(repo, file = %file.path, size = file.size, attempt, "downloading checkpoint file");
        match download_attempt(agent, &url, file, &target) {
            Ok(()) => return Ok(()),
            Err(error) => {
                tracing::warn!(repo, file = %file.path, attempt, error = %error, "download attempt failed");
                last_error = Some(error);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        Error::config(format!("download of `{}` failed after retries", file.path))
    }))
}

/// A cached file matches when its size is right and its SHA-256 vouches for
/// it — directly, or through the sidecar a previous verification wrote.
fn cached_file_matches(target: &Path, file: &SnapshotFile) -> Result<bool, Error> {
    let metadata = match fs::metadata(target) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(Error::Io(io::Error::new(
                error.kind(),
                format!("stat `{}`: {}", target.display(), error),
            )));
        }
    };
    if !metadata.is_file() || metadata.len() != file.size {
        tracing::warn!(path = %target.display(), "cached file has wrong size; redownloading");
        return Ok(false);
    }
    let Some(expected) = &file.sha256 else {
        // Nothing vouches for a hashless cache entry, and these are small
        // config and tokenizer files, so always take them fresh.
        return Ok(false);
    };
    if sidecar_records_hash(target, expected) {
        return Ok(true);
    }
    match hash_file(target) {
        Ok(hash) if hash == *expected => {
            if let Err(error) = write_sidecar(target, expected) {
                tracing::debug!(path = %target.display(), error = %error, "failed to write sha256 sidecar; cache will rehash next time");
            }
            Ok(true)
        }
        Ok(hash) => {
            tracing::warn!(path = %target.display(), expected = %expected, actual = %hash, "cached file sha256 mismatch; redownloading");
            Ok(false)
        }
        Err(error) => {
            tracing::warn!(path = %target.display(), error = %error, "failed to hash cached file; redownloading");
            Ok(false)
        }
    }
}

/// Accepts a rename destination that lost a race to a concurrent download:
/// it matches when its size is right and its published hash (or sidecar)
/// vouches for it — by size alone when the source published no hash.
fn destination_matches(target: &Path, file: &SnapshotFile) -> bool {
    let Ok(metadata) = fs::metadata(target) else {
        return false;
    };
    if !metadata.is_file() || metadata.len() != file.size {
        return false;
    }
    match &file.sha256 {
        None => true,
        Some(expected) => {
            sidecar_records_hash(target, expected)
                || hash_file(target).is_ok_and(|hash| hash == *expected)
        }
    }
}

/// Monotonic counter keeping concurrent downloads of the same file from
/// sharing a temp path; with the PID it is unique without a `rand` dep.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_tmp_path(target: &Path) -> PathBuf {
    let counter = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    target.with_file_name(format!(
        ".{}.{}.{}.part",
        target.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        counter
    ))
}

/// Deletes a temp file on drop unless defused by a successful rename.
struct TempFileGuard {
    path: Option<PathBuf>,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn path(&self) -> &Path {
        self.path.as_deref().expect("guard already disarmed")
    }

    /// Hand the temp file off to a successful rename.
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
    file: &SnapshotFile,
    target: &Path,
) -> Result<(), Error> {
    let response = agent
        .get(url)
        .call()
        .map_err(|error| Error::Io(io::Error::other(format!("GET {url}: {error}"))))?;
    let mut body = response.into_body().into_reader();

    let tmp = unique_tmp_path(target);
    let mut handle = File::create(&tmp).map_err(|error| {
        Error::Io(io::Error::new(
            error.kind(),
            format!("create `{}`: {}", tmp.display(), error),
        ))
    })?;
    // Any early return past this point must not leak the temp file.
    let guard = TempFileGuard::new(tmp);

    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    let mut buffer = vec![0u8; READ_BUFFER_BYTES];
    let mut written: u64 = 0;
    loop {
        let read = body.read(&mut buffer).map_err(|error| {
            Error::Io(io::Error::other(format!(
                "read body for `{}`: {}",
                file.path, error
            )))
        })?;
        if read == 0 {
            break;
        }
        sha2::Digest::update(&mut hasher, &buffer[..read]);
        handle.write_all(&buffer[..read]).map_err(|error| {
            Error::Io(io::Error::new(
                error.kind(),
                format!("write `{}`: {}", guard.path().display(), error),
            ))
        })?;
        written += read as u64;
    }
    handle.sync_all().map_err(|error| {
        Error::Io(io::Error::other(format!(
            "sync `{}`: {}",
            guard.path().display(),
            error
        )))
    })?;
    drop(handle);

    if written != file.size {
        return Err(Error::config(format!(
            "downloaded `{}` is {} bytes but the source lists {}",
            file.path, written, file.size
        )));
    }
    if let Some(expected) = &file.sha256 {
        let actual = encode_hex(&sha2::Digest::finalize(hasher));
        if actual != *expected {
            return Err(Error::config(format!(
                "sha256 mismatch for `{}`: expected {expected}, got {actual}",
                file.path
            )));
        }
    }

    if let Err(error) = fs::rename(guard.path(), target) {
        // A concurrent download may have renamed its copy onto the target
        // between our checks and this rename (notably on Windows); accept
        // the winner when it matches what we asked for.
        if !destination_matches(target, file) {
            return Err(Error::Io(io::Error::new(
                error.kind(),
                format!(
                    "move `{}` -> `{}`: {}",
                    guard.path().display(),
                    target.display(),
                    error
                ),
            )));
        }
        // The guard's drop removes our now-redundant temp file.
        return Ok(());
    }
    guard.disarm();

    if let Some(hash) = &file.sha256
        && let Err(error) = write_sidecar(target, hash)
    {
        tracing::debug!(path = %target.display(), error = %error, "failed to write sha256 sidecar after download");
    }
    Ok(())
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
    fs::write(sidecar, hash)
}

fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    let mut buffer = vec![0u8; READ_BUFFER_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        sha2::Digest::update(&mut hasher, &buffer[..read]);
    }
    Ok(encode_hex(&sha2::Digest::finalize(hasher)))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_dirs_follow_org_and_name() {
        let root = Path::new("/cache");
        assert_eq!(
            snapshot_dir(root, "PaddlePaddle/PaddleOCR-VL-1.5").unwrap(),
            root.join("models/PaddlePaddle/PaddleOCR-VL-1.5")
        );
        assert_eq!(
            snapshot_dir(root, "opendatalab/MinerU2.5-2509-1.2B").unwrap(),
            root.join("models/opendatalab/MinerU2.5-2509-1.2B")
        );
        assert!(snapshot_dir(root, "no-slash").is_err());
    }

    #[test]
    fn file_listings_parse_hashes_and_skip_directories() {
        // ModelScope publishes a SHA-256 for every blob.
        let listing = r#"{"Code":200,"Data":{"Files":[
            {"Path":".gitattributes","Sha256":"443f","Size":2082,"Type":"blob"},
            {"Path":"model.safetensors","Sha256":"5ea4","Size":133270468,"Type":"blob"},
            {"Path":"subdir","Sha256":null,"Size":0,"Type":"tree"}
        ]}}"#;
        assert_eq!(
            parse_modelscope_listing(listing).unwrap(),
            vec![
                SnapshotFile {
                    path: ".gitattributes".into(),
                    size: 2082,
                    sha256: Some("443f".into())
                },
                SnapshotFile {
                    path: "model.safetensors".into(),
                    size: 133_270_468,
                    sha256: Some("5ea4".into())
                },
            ]
        );
        // Hugging Face publishes a SHA-256 only for LFS entries.
        let listing = r#"[
            {"path":"config.json","size":2460,"type":"file"},
            {"path":"model.safetensors","size":133270468,"lfs":{"oid":"5ea4","size":133270468},"type":"file"},
            {"path":"subdir","type":"directory"}
        ]"#;
        assert_eq!(
            parse_huggingface_listing(listing).unwrap(),
            vec![
                SnapshotFile {
                    path: "config.json".into(),
                    size: 2460,
                    sha256: None
                },
                SnapshotFile {
                    path: "model.safetensors".into(),
                    size: 133_270_468,
                    sha256: Some("5ea4".into())
                },
            ]
        );
    }

    #[test]
    fn prune_removes_unlisted_files_and_keeps_bookkeeping() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("config.json"), b"{}").unwrap();
        std::fs::write(root.join("stale.bin"), b"x").unwrap();
        std::fs::create_dir(root.join("v1.0")).unwrap();
        std::fs::write(root.join("v1.0/stale.onnx"), b"x").unwrap();
        std::fs::write(root.join(".stale.bin.sha256"), b"hash").unwrap();
        std::fs::write(root.join(".oar-revision"), b"repo\nmaster\n").unwrap();
        let files = [SnapshotFile {
            path: "config.json".to_string(),
            size: 2,
            sha256: None,
        }];
        prune_stale_files(root, &files).unwrap();
        assert!(root.join("config.json").exists());
        assert!(root.join(".stale.bin.sha256").exists());
        assert!(root.join(".oar-revision").exists());
        assert!(!root.join("stale.bin").exists());
        // The emptied v1.0 directory went with its file.
        assert!(!root.join("v1.0").exists());

        // URL components and Link pages, shared here to keep tests lean.
        assert_eq!(encode_component("refs/pr 1#2"), "refs%2Fpr%201%232");
        assert_eq!(encode_path("a b/c#d"), "a%20b/c%23d");
        let link = r#"<https://huggingface.co/api/models/m/tree/main?recursive=true&page=2>; rel="next", <https://huggingface.co/api/models/m/tree/main?recursive=true&page=1>; rel="prev""#;
        assert_eq!(
            next_page(link),
            Some("https://huggingface.co/api/models/m/tree/main?recursive=true&page=2")
        );
        assert_eq!(next_page(r#"<https://x?page=1>; rel="prev""#), None);
    }

    #[test]
    fn modelscope_selection_resolves_aliases_and_falls_back() {
        // The verified mirror is used for the aliased repo.
        assert_eq!(
            resolve_remote(DownloadSource::ModelScope, "zai-org/GLM-OCR"),
            (DownloadSource::ModelScope, "ZhipuAI/GLM-OCR".to_string())
        );
        // Repos ModelScope lacks download from Hugging Face instead.
        for repo in [
            "tencent/HunyuanOCR",
            "tencent/WeVisDoc-2B",
            "tencent/WeVisDoc-4B",
        ] {
            assert_eq!(
                resolve_remote(DownloadSource::ModelScope, repo),
                (DownloadSource::HuggingFace, repo.to_string())
            );
        }
        // Everything else passes through, as does an explicit HF choice.
        assert_eq!(
            resolve_remote(DownloadSource::ModelScope, "PaddlePaddle/HPD-Parsing"),
            (
                DownloadSource::ModelScope,
                "PaddlePaddle/HPD-Parsing".to_string()
            )
        );
        assert_eq!(
            resolve_remote(DownloadSource::HuggingFace, "zai-org/GLM-OCR"),
            (DownloadSource::HuggingFace, "zai-org/GLM-OCR".to_string())
        );
    }
}
