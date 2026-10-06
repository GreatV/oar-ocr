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
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Default layout checkpoint downloaded for layout-composed models.
pub const DEFAULT_LAYOUT_REPO: &str = "PaddlePaddle/PP-DocLayoutV3_safetensors";

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
                "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision={revision}&Recursive=true"
            ),
            Self::HuggingFace => {
                format!("https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true")
            }
        }
    }

    fn file_url(self, repo: &str, revision: &str, path: &str) -> String {
        match self {
            Self::ModelScope => format!(
                "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision={revision}&FilePath={path}"
            ),
            Self::HuggingFace => format!("https://huggingface.co/{repo}/resolve/{revision}/{path}"),
        }
    }
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

    /// Pin a specific revision on both the model and layout repos.
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

/// Downloads (or reuses) the pinned snapshot of `repo` and returns its
/// directory in the cache.
pub(crate) fn snapshot(
    source: DownloadSource,
    repo: &str,
    revision: Option<&str>,
) -> Result<PathBuf, Error> {
    let revision = revision.unwrap_or_else(|| source.default_revision());
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

    let files = list_files(&agent, source, repo, revision)?;
    for file in &files {
        ensure_file(&agent, source, repo, revision, &dir, file)?;
    }

    let marker = dir.join(".oar-revision");
    let current = format!("{repo}\n{revision}\n");
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

/// Lists a repo's files through the source API.
fn list_files(
    agent: &ureq::Agent,
    source: DownloadSource,
    repo: &str,
    revision: &str,
) -> Result<Vec<SnapshotFile>, Error> {
    let url = source.files_url(repo, revision);
    let response = agent
        .get(&url)
        .call()
        .map_err(|error| listing_error(source, repo, error))?;
    let body = response
        .into_body()
        .read_to_string()
        .map_err(|error| Error::Io(io::Error::other(format!("read {url}: {error}"))))?;
    match source {
        DownloadSource::ModelScope => parse_modelscope_listing(&body),
        DownloadSource::HuggingFace => parse_huggingface_listing(&body),
    }
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
        // The source published no hash; size is all we can check.
        return Ok(true);
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

    fs::rename(guard.path(), target).map_err(|error| {
        Error::Io(io::Error::new(
            error.kind(),
            format!(
                "move `{}` -> `{}`: {}",
                guard.path().display(),
                target.display(),
                error
            ),
        ))
    })?;
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
    fn pretrained_defaults_download_modelscope_layout_v3() {
        assert_eq!(DownloadSource::ModelScope.default_revision(), "master");
        assert_eq!(DownloadSource::HuggingFace.default_revision(), "main");
        let options = AnyPageParserPretrainedOptions::default();
        assert_eq!(options.source(), DownloadSource::ModelScope);
        assert!(options.revision.is_none());
        assert_eq!(options.layout(), DEFAULT_LAYOUT_REPO);
        let options = options
            .with_source(DownloadSource::HuggingFace)
            .with_revision("v1.6.0")
            .with_layout("PaddlePaddle/PP-DocLayoutV2_safetensors");
        assert_eq!(options.source(), DownloadSource::HuggingFace);
        assert_eq!(options.revision.as_deref(), Some("v1.6.0"));
        assert_eq!(options.layout(), "PaddlePaddle/PP-DocLayoutV2_safetensors");
    }
}
