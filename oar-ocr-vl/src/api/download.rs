//! Snapshot download of VL checkpoints (feature `auto-download`).
//!
//! [`AnyPageParser::from_pretrained`](crate::AnyPageParser::from_pretrained)
//! downloads a checkpoint repo by its [`AnyPageParserModel`](crate::AnyPageParserModel)
//! ID when it is not cached, then loads it through the regular directory
//! path. The requested revision is resolved to an immutable commit first,
//! and each snapshot lives at `$OAR_HOME/models/<org>/<name>/<commit>`
//! (`$OAR_HOME` defaults to `~/.oar`, as in oar-ocr-core). A snapshot is
//! downloaded into a `<commit>.partial` staging directory, every file is
//! verified against the hash the source API provides (ModelScope always
//! publishes SHA-256; Hugging Face publishes it for LFS files), and the
//! staging directory is renamed into place only once complete. Published
//! snapshots are never modified again, so loading one needs no lock and a
//! failed download leaves earlier snapshots untouched. A
//! `refs/<source>/<revision>` file records which commit each requested
//! revision resolved to on each source.

use crate::api::error::Error;
use serde::Deserialize;
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
    /// Directory under `refs/` that records this source's resolutions.
    fn ref_dir(self) -> &'static str {
        match self {
            Self::ModelScope => "modelscope",
            Self::HuggingFace => "huggingface",
        }
    }

    /// The revision downloaded when none is pinned in the options.
    fn default_revision(self) -> &'static str {
        match self {
            Self::ModelScope => "master",
            Self::HuggingFace => "main",
        }
    }

    /// Resolves a revision (branch, tag, or commit) to its commit id.
    fn resolve_url(self, repo: &str, revision: &str) -> String {
        match self {
            // ModelScope's commits endpoint resolves the revision through
            // `Ref`; a `Revision` parameter is silently ignored there and
            // always yields the default branch's head.
            Self::ModelScope => format!(
                "https://www.modelscope.cn/api/v1/models/{}/commits?Ref={}&PageSize=1",
                encode_path(repo),
                encode_component(revision)
            ),
            Self::HuggingFace => format!(
                "https://huggingface.co/api/models/{}/revision/{}",
                encode_path(repo),
                encode_component(revision)
            ),
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
                encode_component(revision)
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
                encode_component(revision),
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

#[derive(Deserialize)]
struct ModelScopeCommits {
    #[serde(rename = "Data")]
    data: ModelScopeCommitData,
}

#[derive(Deserialize)]
struct ModelScopeCommitData {
    #[serde(rename = "Commit")]
    commits: Vec<ModelScopeCommit>,
}

#[derive(Deserialize)]
struct ModelScopeCommit {
    #[serde(rename = "Id")]
    id: String,
}

#[derive(Deserialize)]
struct HuggingFaceRevision {
    sha: String,
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

/// Parses a ModelScope `repo/files` listing into snapshot files, plus the
/// raw entry count including the tree entries that are filtered out — the
/// endpoint's truncation applies to raw entries.
fn parse_modelscope_listing(body: &str) -> Result<(Vec<SnapshotFile>, usize), Error> {
    let listing: ModelScopeListing = serde_json::from_str(body)
        .map_err(|error| Error::config(format!("parse ModelScope file listing: {error}")))?;
    let raw_entries = listing.data.files.len();
    let files = listing
        .data
        .files
        .into_iter()
        .filter(|entry| entry.kind == "blob")
        .map(|entry| SnapshotFile {
            path: entry.path,
            size: entry.size,
            sha256: entry.sha256,
        })
        .collect();
    Ok((files, raw_entries))
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

/// Maps a repo ID to the directory holding its commit snapshots.
fn snapshot_base(root: &Path, repo: &str) -> Result<PathBuf, Error> {
    validate_repo_id(repo)?;
    let (org, name) = repo.split_once('/').expect("validated ids split once");
    Ok(root.join("models").join(org).join(name))
}

/// Commit ids are hex strings from the source APIs; anything else would be
/// joined into the cache path unchecked.
fn validate_commit(commit: &str) -> Result<(), Error> {
    let hex =
        (40..=64).contains(&commit.len()) && commit.bytes().all(|byte| byte.is_ascii_hexdigit());
    hex.then_some(())
        .ok_or_else(|| Error::config(format!("source returned a non-hex commit id {commit:?}")))
}

/// Validates a repository id before it becomes a filesystem path: exactly
/// two non-empty `<org>/<name>` segments, neither `.` nor `..`, no
/// backslash, and not absolute. Anything else could escape the cache
/// directory — the prune walk would follow it out of `$OAR_HOME/models`.
fn validate_repo_id(repo: &str) -> Result<(), Error> {
    let invalid = || {
        Error::config(format!(
            "model id {repo:?} is not an <org>/<name> repository id"
        ))
    };
    if repo.is_empty() || repo.contains('\\') || repo.starts_with('/') {
        return Err(invalid());
    }
    let mut segments = repo.split('/');
    match [segments.next(), segments.next(), segments.next()] {
        [Some(org), Some(name), None]
            if !org.is_empty()
                && !name.is_empty()
                && !matches!(org, "." | "..")
                && !matches!(name, "." | "..") =>
        {
            // Same component rules as listed file paths, so a Windows drive
            // prefix (`C:/payload`) fails on every platform before any
            // directory is created.
            if segments_contain_colon(repo) || !components_are_normal(repo) {
                return Err(invalid());
            }
            Ok(())
        }
        _ => Err(invalid()),
    }
}

/// Whether any `/`-separated segment contains a colon (Windows drive
/// prefixes and NTFS stream names).
fn segments_contain_colon(path: &str) -> bool {
    path.split('/').any(|segment| segment.contains(':'))
}

/// Whether every path component is [`std::path::Component::Normal`], i.e.
/// nothing [`Path::join`] would treat specially.
fn components_are_normal(path: &str) -> bool {
    Path::new(path)
        .components()
        .all(|component| matches!(component, std::path::Component::Normal(_)))
}

/// ModelScope's `repo/files` endpoint silently truncates at this many
/// entries.
const MODELSCOPE_LISTING_LIMIT: usize = 3_000;
const DOWNLOAD_RETRIES: u32 = 3;
const READ_BUFFER_BYTES: usize = 64 * 1024;
const CONNECT_TIMEOUT_SECS: u64 = 30;
/// How long to wait for response headers; body reads are unbounded so
/// multi-GB shards survive slow links.
const RESPONSE_HEADER_TIMEOUT_SECS: u64 = 60;

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

/// Downloads (or reuses) the snapshot of `repo` at `revision` and returns
/// its directory in the cache.
///
/// The revision resolves to an immutable commit first; a published
/// `<commit>/` snapshot is returned as-is, and a missing one is staged in
/// `<commit>.partial` next to it and renamed into place only once every file
/// is downloaded and verified. The per-repo lock serializes staging and
/// publishing across processes; loading a published snapshot needs no lock.
pub(crate) fn snapshot(
    source: DownloadSource,
    repo: &str,
    revision: Option<&str>,
) -> Result<PathBuf, Error> {
    let (source, remote) = resolve_remote(source, repo);
    let revision = revision.unwrap_or_else(|| source.default_revision());
    let base = snapshot_base(&cache_root(), repo)?;
    fs::create_dir_all(&base).map_err(|error| {
        Error::Io(io::Error::new(
            error.kind(),
            format!("create snapshot directory `{}`: {}", base.display(), error),
        ))
    })?;

    let agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(CONNECT_TIMEOUT_SECS)))
        .timeout_recv_response(Some(Duration::from_secs(RESPONSE_HEADER_TIMEOUT_SECS)))
        .build()
        .new_agent();

    // A requested revision that already is a commit id with a local snapshot
    // needs no resolution at all.
    if validate_commit(revision).is_ok() && base.join(revision).is_dir() {
        record_commit(&base, source, revision, revision);
        return Ok(base.join(revision));
    }
    let commit = match resolve_commit(&agent, source, &remote, revision) {
        Ok(commit) => commit,
        Err(error) => {
            // Offline (or hub) failure: fall back to the commit this
            // revision resolved to before, when its snapshot survived.
            if let Some(recorded) = recorded_commit(&base, source, revision)
                && base.join(&recorded).is_dir()
            {
                tracing::warn!(
                    repo,
                    revision,
                    commit = %recorded,
                    "revision resolution failed; reusing the recorded snapshot"
                );
                return Ok(base.join(recorded));
            }
            return Err(error);
        }
    };
    let dir = base.join(&commit);
    if dir.is_dir() {
        record_commit(&base, source, revision, &commit);
        return Ok(dir);
    }
    let _lock = SnapshotLock::acquire(&base)?;
    // Another process may have published while we waited for the lock.
    if dir.is_dir() {
        record_commit(&base, source, revision, &commit);
        return Ok(dir);
    }
    let staging = base.join(format!("{commit}.partial"));
    // A crashed attempt's staging is incomplete by definition; start over.
    fs::remove_dir_all(&staging)
        .or_else(|error| match error.kind() {
            io::ErrorKind::NotFound => Ok(()),
            _ => Err(error),
        })
        .map_err(|error| {
            Error::Io(io::Error::new(
                error.kind(),
                format!("clean staging `{}`: {}", staging.display(), error),
            ))
        })?;
    fs::create_dir_all(&staging).map_err(|error| {
        Error::Io(io::Error::new(
            error.kind(),
            format!("create staging `{}`: {}", staging.display(), error),
        ))
    })?;

    let files = list_files(&agent, source, &remote, &commit)?;
    for file in &files {
        ensure_file(&agent, source, &remote, &commit, &staging, file)?;
    }
    fs::rename(&staging, &dir).map_err(|error| {
        Error::Io(io::Error::new(
            error.kind(),
            format!(
                "publish `{}` -> `{}`: {}",
                staging.display(),
                dir.display(),
                error
            ),
        ))
    })?;

    record_commit(&base, source, revision, &commit);
    Ok(dir)
}

/// Reads the commit a revision last resolved to on this source, when
/// `refs/<source>/<revision>` recorded one.
fn recorded_commit(base: &Path, source: DownloadSource, revision: &str) -> Option<String> {
    let path = base
        .join("refs")
        .join(source.ref_dir())
        .join(encode_component(revision));
    let recorded = fs::read_to_string(path).ok()?;
    let commit = recorded.trim();
    validate_commit(commit).is_ok().then(|| commit.to_string())
}

/// Best-effort record of what a revision resolved to on this source, so the
/// latest completed snapshot can be found offline.
fn record_commit(base: &Path, source: DownloadSource, revision: &str, commit: &str) {
    let refs = base.join("refs").join(source.ref_dir());
    if let Err(error) = fs::create_dir_all(&refs)
        .and_then(|()| fs::write(refs.join(encode_component(revision)), commit))
    {
        tracing::debug!(error = %error, "failed to record the resolved revision");
    }
}

/// Resolves a revision to the immutable commit id both the listing and every
/// file download are then pinned to.
fn resolve_commit(
    agent: &ureq::Agent,
    source: DownloadSource,
    repo: &str,
    revision: &str,
) -> Result<String, Error> {
    let url = source.resolve_url(repo, revision);
    let response = agent.get(&url).call().map_err(|error| {
        let base = format!("resolve {repo} revision {revision} on {source:?}: {error}");
        if matches!(&error, ureq::Error::StatusCode(404)) {
            Error::config(format!(
                "{base}; the repository or revision is not published on the selected source — \
                 try DownloadSource::HuggingFace or DownloadSource::ModelScope"
            ))
        } else {
            Error::Io(io::Error::other(base))
        }
    })?;
    let body = response
        .into_body()
        .read_to_string()
        .map_err(|error| Error::Io(io::Error::other(format!("read {url}: {error}"))))?;
    let commit = match source {
        DownloadSource::ModelScope => {
            let commits: ModelScopeCommits = serde_json::from_str(&body)
                .map_err(|error| Error::config(format!("parse ModelScope commits: {error}")))?;
            commits
                .data
                .commits
                .into_iter()
                .next()
                .map(|commit| commit.id)
                .ok_or_else(|| Error::config("ModelScope returned no commit for the revision"))?
        }
        DownloadSource::HuggingFace => {
            let revision: HuggingFaceRevision = serde_json::from_str(&body)
                .map_err(|error| Error::config(format!("parse Hugging Face revision: {error}")))?;
            revision.sha
        }
    };
    validate_commit(&commit)?;
    Ok(commit)
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
        let page = match source {
            DownloadSource::ModelScope => {
                let (files, raw_entries) = parse_modelscope_listing(&body)?;
                // ModelScope truncates `repo/files` listings at 3,000 raw
                // entries with no marker; refuse to publish a snapshot from
                // a possibly partial listing rather than trust it.
                if raw_entries >= MODELSCOPE_LISTING_LIMIT {
                    return Err(Error::config(format!(
                        "ModelScope listing for {repo} returned {raw_entries} entries and may \
                         be truncated; refusing to publish an incomplete snapshot"
                    )));
                }
                files
            }
            DownloadSource::HuggingFace => parse_huggingface_listing(&body)?,
        };
        files.extend(page);
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

/// Downloads one listed file into the staging directory, verifying it
/// against the published hash and size.
fn ensure_file(
    agent: &ureq::Agent,
    source: DownloadSource,
    repo: &str,
    revision: &str,
    staging: &Path,
    file: &SnapshotFile,
) -> Result<(), Error> {
    validate_listed_path(&file.path)?;
    let target = staging.join(&file.path);
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

/// Validates a file path from a listing before it is joined into the staging
/// directory: no absolute paths, no `.` or `..` segments, no backslashes, no
/// empty segments, no Windows drive prefixes, and nothing else
/// [`Path::join`] would treat specially — every component must be
/// [`std::path::Component::Normal`].
fn validate_listed_path(path: &str) -> Result<(), Error> {
    let invalid = || {
        Error::config(format!(
            "listed file path {path:?} is not relative and simple"
        ))
    };
    if path.is_empty() || path.starts_with('/') || path.contains('\\') {
        return Err(invalid());
    }
    for segment in path.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(invalid());
        }
    }
    // The colon check rejects Windows drive prefixes (`C:/payload`)
    // everywhere; the component check rejects anything else Path::join
    // would treat specially.
    if segments_contain_colon(path) || !components_are_normal(path) {
        return Err(invalid());
    }
    Ok(())
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

/// Holds an exclusive advisory lock on `<snapshot>.lock` for the duration
/// of a snapshot sync; dropping it releases the lock. No timeouts and no
/// stale-lock recovery — a crashed holder simply leaves an unlocked file.
struct SnapshotLock {
    file: File,
}

impl SnapshotLock {
    fn acquire(dir: &Path) -> Result<Self, Error> {
        let name = dir.file_name().unwrap_or_default().to_string_lossy();
        let path = dir.with_file_name(format!("{name}.lock"));
        let file = File::create(&path).map_err(|error| {
            Error::Io(io::Error::new(
                error.kind(),
                format!("create `{}`: {}", path.display(), error),
            ))
        })?;
        file.lock().map_err(|error| {
            Error::Io(io::Error::other(format!(
                "lock `{}`: {}",
                path.display(),
                error
            )))
        })?;
        Ok(Self { file })
    }
}

impl Drop for SnapshotLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
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
    Ok(())
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
    fn snapshot_bases_and_ids_are_validated() {
        let root = Path::new("/cache");
        assert_eq!(
            snapshot_base(root, "PaddlePaddle/PaddleOCR-VL-1.5").unwrap(),
            root.join("models/PaddlePaddle/PaddleOCR-VL-1.5")
        );
        // Commits land one level below the base, hex only.
        let base = snapshot_base(root, "opendatalab/MinerU2.5-2509-1.2B").unwrap();
        assert_eq!(
            base.join("74851b0e4989e661958cd2de23a4a1e3ef4fd5a9"),
            root.join(
                "models/opendatalab/MinerU2.5-2509-1.2B/74851b0e4989e661958cd2de23a4a1e3ef4fd5a9"
            )
        );
        assert!(validate_commit("74851b0e4989e661958cd2de23a4a1e3ef4fd5a9").is_ok());
        for bad_commit in ["", "xyz", "../escape", "74851b0e"] {
            assert!(validate_commit(bad_commit).is_err(), "{bad_commit:?}");
        }
        // Ids that could escape the cache directory are rejected outright.
        for bad in [
            "no-slash",
            "/absolute/repo",
            "org/../neighbor",
            "..",
            r"org/name\with\backslash",
            "org//name",
            "org/name/extra",
            "org/.",
            "C:/payload",
            "org/na:me",
            "",
        ] {
            assert!(snapshot_base(root, bad).is_err(), "{bad:?}");
        }
        // Listed file paths that could escape the staging directory too.
        for bad in [
            "/absolute/file",
            "../escape",
            "a/./b",
            "a//b",
            "trailing/",
            "a/../b",
            r"a\b",
            "C:/Users/Public/payload",
            "drive:relative",
            "",
        ] {
            assert!(validate_listed_path(bad).is_err(), "{bad:?}");
        }
        assert!(validate_listed_path("config.json").is_ok());
        assert!(validate_listed_path("v1.0/model-00001-of-00004.safetensors").is_ok());
    }

    #[test]
    fn file_listings_parse_hashes_and_skip_directories() {
        // ModelScope publishes a SHA-256 for every blob.
        let listing = r#"{"Code":200,"Data":{"Files":[
            {"Path":".gitattributes","Sha256":"443f","Size":2082,"Type":"blob"},
            {"Path":"model.safetensors","Sha256":"5ea4","Size":133270468,"Type":"blob"},
            {"Path":"subdir","Sha256":null,"Size":0,"Type":"tree"}
        ]}}"#;
        let (files, raw_entries) = parse_modelscope_listing(listing).unwrap();
        assert_eq!(
            files,
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
        // The raw count includes the tree entry the files list drops.
        assert_eq!(raw_entries, 3);
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
    fn urls_encode_revisions_as_one_component_and_pages_parse() {
        assert_eq!(
            DownloadSource::HuggingFace.files_url("org/name", "refs/pr 1"),
            "https://huggingface.co/api/models/org/name/tree/refs%2Fpr%201?recursive=true"
        );
        assert_eq!(
            DownloadSource::HuggingFace.resolve_url("org/name", "main"),
            "https://huggingface.co/api/models/org/name/revision/main"
        );
        assert_eq!(
            DownloadSource::ModelScope.resolve_url("org/name", "master"),
            "https://www.modelscope.cn/api/v1/models/org/name/commits?Ref=master&PageSize=1"
        );
        assert_eq!(
            DownloadSource::ModelScope.file_url("org/name", "abc123", "a b/c#d.bin"),
            "https://www.modelscope.cn/api/v1/models/org/name/repo?Revision=abc123&FilePath=a%20b/c%23d.bin"
        );
        let link = r#"<https://huggingface.co/api/models/m/tree/main?recursive=true&page=2>; rel="next", <https://huggingface.co/api/models/m/tree/main?recursive=true?page=1>; rel="prev""#;
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
