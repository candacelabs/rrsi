// Copyright 2026 Candace Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Toolchains: what "the tests of this change" and "a test run" mean for one
//! build system. A [`Toolchain`] owns five things:
//!
//! 1. detect project roots (`go.mod`, `pyproject.toml`, `CMakeLists.txt`,
//!    `MODULE.bazel`, ...);
//! 2. classify every changed file as source, test, generated or other;
//! 3. map a commit's changed tests to test units (Go package, pytest file,
//!    CTest test, Bazel test target);
//! 4. run those units offline in a pinned container, after a separate
//!    networked step that prefetches the dependencies (Go's `download`);
//! 5. classify a run as Pass, TestFail, BuildFail, Infra or Timeout.
//!
//! Go ([`go::Go`]) is one implementation and behaves exactly as the miner
//! did before toolchains existed (tests/go_list_parity.rs pins it). The
//! FAIL_TO_PASS rule ([`crate::decide`]) and the task layout are shared.

use crate::{dir_of, git, Candidate, Outcome, MAX_SRC_CHURN, MAX_SRC_PACKAGES};
use anyhow::{bail, Context, Result};
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

pub mod bazel;
pub mod cpp;
pub mod go;
pub mod python;

/// What a changed file is, for one toolchain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileKind {
    /// Part of the fix: in src.patch and the churn.
    Source,
    /// Part of the hidden tests: in tests.patch, applied to the parent.
    Test,
    /// Generated from other files: never part of a fix.
    Generated,
    /// Neither (docs, other languages): ignored.
    Other,
}

/// One build system the miner can validate tasks with.
pub trait Toolchain: Sync {
    /// `go`, `python`, `cpp` or `bazel`: the value of `--toolchain` and of a
    /// task's `toolchain` field.
    fn name(&self) -> &'static str;
    /// Whether a repository-relative path marks a project root.
    fn is_root_marker(&self, path: &str) -> bool;
    fn classify_file(&self, path: &str) -> FileKind;
    /// The candidate commits since `since`, newest first.
    fn candidates(&self, repo: &Path, since: &str) -> Result<Vec<Candidate>>;
    /// The units to run, resolved on the commit tree after its prefetch
    /// (a CTest test name, a Bazel test label). An empty list means no test
    /// unit covers the changed tests.
    fn resolve_units(&self, _commit_tree: &Path, cand: &Candidate) -> Result<Vec<String>> {
        Ok(cand.packages.clone())
    }
    /// The progress label of the networked step, e.g. `go mod download`.
    fn prefetch_label(&self) -> &'static str;
    /// The progress label of a test run of `units`, e.g. `go test ./pkg`.
    fn test_label(&self, units: &[String]) -> String;
    /// With network: fetch everything `tree` needs so the offline run can
    /// build. A failure is recorded but is not a verdict by itself.
    fn prefetch(&self, tree: &Path, cand: &Candidate) -> Result<(bool, String)>;
    /// Without network: run the candidate's units on `tree`, classified.
    fn run_tests(&self, tree: &Path, cand: &Candidate) -> Result<(Outcome, String)>;

    /// The project roots at `rev`: directories holding a root marker.
    fn detect(&self, repo: &Path, rev: &str) -> Result<Vec<String>> {
        let files = tracked_files(repo, rev)?;
        Ok(files.iter().filter(|f| self.is_root_marker(f)).map(|f| dir_of(f))
            .collect::<BTreeSet<_>>().into_iter().collect())
    }
}

/// Every toolchain, in the order `--toolchain auto` tries them: a commit is
/// claimed by the first toolchain that makes it a candidate.
pub const NAMES: [&str; 4] = ["go", "python", "cpp", "bazel"];

/// The language a toolchain's tasks are written in, for prompts: (name in
/// prose, code-fence tag). A task without a toolchain is Go.
pub fn language(toolchain: Option<&str>) -> (&'static str, &'static str) {
    match toolchain.unwrap_or("go") {
        "python" => ("Python", "python"),
        "cpp" => ("C++", "cpp"),
        "bazel" => ("Bazel-built", ""),
        _ => ("Go", "go"),
    }
}

/// How the named toolchain (absent: Go) classifies a path, without its
/// container settings.
pub fn classify_file(toolchain: Option<&str>, path: &str) -> FileKind {
    match toolchain.unwrap_or("go") {
        "python" => python::classify_file(path),
        "cpp" => cpp::classify_file(path),
        "bazel" => bazel::classify_file(path),
        _ => go::classify_file(path),
    }
}

/// The toolchains one command uses, looked up by a candidate's
/// `toolchain` field (absent means Go, as in every task mined before).
pub struct Toolchains {
    pub all: Vec<Box<dyn Toolchain>>,
}

impl Toolchains {
    pub fn get(&self, name: Option<&str>) -> Result<&dyn Toolchain> {
        let name = name.unwrap_or("go");
        self.all.iter().find(|t| t.name() == name).map(|b| b.as_ref())
            .with_context(|| format!("toolchain {name:?} is not configured"))
    }

    pub fn for_candidate(&self, c: &Candidate) -> Result<&dyn Toolchain> {
        self.get(c.toolchain.as_deref())
    }

    /// The toolchains `--toolchain` selects for `repo`: one by name, or for
    /// `auto` every toolchain with a project root at HEAD, in [`NAMES`] order.
    pub fn select(&self, repo: &Path, choice: &str) -> Result<Vec<&dyn Toolchain>> {
        if choice != "auto" {
            return Ok(vec![self.get(Some(choice))?]);
        }
        let mut out = Vec::new();
        for n in NAMES {
            let t = self.get(Some(n))?;
            if !t.detect(repo, "HEAD")?.is_empty() {
                out.push(t);
            }
        }
        Ok(out)
    }

    /// The candidates of every selected toolchain, newest first per
    /// toolchain; a commit two toolchains both claim goes to the first.
    pub fn candidates(&self, repo: &Path, since: &str, choice: &str) -> Result<Vec<Candidate>> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for t in self.select(repo, choice)? {
            for c in t.candidates(repo, since)? {
                if seen.insert(c.sha.clone()) {
                    out.push(c);
                }
            }
        }
        Ok(out)
    }
}

// ---- shared candidate scan for the non-Go toolchains ----

/// Repository-relative paths of every file tracked at `rev`.
pub fn tracked_files(repo: &Path, rev: &str) -> Result<Vec<String>> {
    Ok(git(repo, &["ls-tree", "-r", "--name-only", "--full-tree", rev])?
        .lines().map(str::to_string).collect())
}

/// Whether `path` lies in `root` (the empty root is the whole repository).
pub fn under(root: &str, path: &str) -> bool {
    root.is_empty() || path == root || path.starts_with(&format!("{root}/"))
}

/// The path relative to `root`.
pub fn rel<'a>(root: &str, path: &'a str) -> &'a str {
    if root.is_empty() { path } else { path.strip_prefix(root).unwrap_or(path).trim_start_matches('/') }
}

/// Ancestor directories of `dir`, nearest first, ending with "" (the root).
pub fn ancestors(dir: &str) -> Vec<String> {
    let parts: Vec<&str> = if dir.is_empty() { vec![] } else { dir.split('/').collect() };
    (0..=parts.len()).rev().map(|i| parts[..i].join("/")).collect()
}

/// Whether any directory segment of `path` (not its file name) is one of
/// `names`.
pub fn in_dir_named(path: &str, names: &[&str]) -> bool {
    let d = dir_of(path);
    !d.is_empty() && d.split('/').any(|s| names.contains(&s))
}

pub fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

pub fn extension(path: &str) -> &str {
    let name = file_name(path);
    name.rsplit_once('.').map(|(_, e)| e).unwrap_or("")
}

/// One commit's changed files, split by kind, with the files tracked at it.
pub struct Changed {
    pub sha: String,
    /// (churn, path) of the source files.
    pub src: Vec<(u64, String)>,
    pub tests: Vec<String>,
    /// Every file tracked at the commit (a deleted file is absent).
    pub tracked: BTreeSet<String>,
}

/// How one non-Go toolchain turns a commit's changes into a candidate.
pub trait Mapper {
    /// The project root the changed tests belong to, if exactly one.
    fn root(&self, ch: &Changed) -> Option<String>;
    /// The test units of the changed tests under `root`, relative to it.
    fn units(&self, ch: &Changed, root: &str, repo: &Path) -> Result<Vec<String>>;
}

/// The candidates of a non-Go toolchain: commits since `since` that change
/// source and tests of one project root, within the Go miner's size limits
/// (at most MAX_SRC_CHURN changed source lines in at most MAX_SRC_PACKAGES
/// directories), whose changed tests map to at least one test unit.
pub fn scan_candidates(tc: &dyn Toolchain, mapper: &dyn Mapper, repo: &Path, since: &str)
    -> Result<Vec<Candidate>> {
    let since_arg = format!("--since={since}");
    let log = git(repo, &["log", &since_arg, "--no-merges", "--format=%H"])?;
    let mut out = Vec::new();
    for sha in log.split_whitespace() {
        let mut src: Vec<(u64, String)> = Vec::new();
        let mut tests: Vec<String> = Vec::new();
        for line in git(repo, &["show", "--numstat", "--format=", "--no-renames", sha])?.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 3 {
                continue;
            }
            let churn = f[0].parse::<u64>().unwrap_or(0) + f[1].parse::<u64>().unwrap_or(0);
            match tc.classify_file(f[2]) {
                FileKind::Source => src.push((churn, f[2].to_string())),
                FileKind::Test => tests.push(f[2].to_string()),
                FileKind::Generated | FileKind::Other => {}
            }
        }
        if src.is_empty() || tests.is_empty() {
            continue;
        }
        let tracked: BTreeSet<String> = tracked_files(repo, sha)?.into_iter().collect();
        let ch = Changed { sha: sha.to_string(), src, tests, tracked };
        let Some(root) = mapper.root(&ch) else { continue };
        // Every changed source file must belong to the same project.
        if ch.src.iter().any(|(_, f)| !under(&root, f)) || ch.tests.iter().any(|f| !under(&root, f)) {
            continue;
        }
        let churn: u64 = ch.src.iter().map(|(c, _)| c).sum();
        let dirs: BTreeSet<String> = ch.src.iter().map(|(_, f)| dir_of(f)).collect();
        if churn > MAX_SRC_CHURN || dirs.len() > MAX_SRC_PACKAGES {
            continue;
        }
        let units = mapper.units(&ch, &root, repo)?;
        if units.is_empty() {
            continue;
        }
        out.push(Candidate {
            sha: sha.to_string(),
            parent: git(repo, &["rev-parse", &format!("{sha}^")])?.trim().to_string(),
            subject: git(repo, &["log", "-1", "--format=%s", sha])?.trim().to_string(),
            body: git(repo, &["log", "-1", "--format=%b", sha])?.trim().to_string(),
            module_root: root,
            packages: units,
            src_files: ch.src.into_iter().map(|(_, f)| f).collect(),
            test_files: ch.tests,
            src_churn: churn,
            toolchain: Some(tc.name().to_string()),
        });
    }
    Ok(out)
}

/// The single nearest-root of `files` (nearest ancestor holding a marker in
/// `tracked`), if they all share one; `topmost` takes the outermost marker
/// instead (CMake: the top-level project owns the CTest tree).
pub fn single_root(files: &[String], tracked: &BTreeSet<String>, is_marker: impl Fn(&str) -> bool,
                   topmost: bool) -> Option<String> {
    let markers: BTreeSet<String> = tracked.iter().filter(|f| is_marker(f)).map(|f| dir_of(f)).collect();
    let roots: BTreeSet<Option<String>> = files.iter().map(|f| {
        let mut a = ancestors(&dir_of(f));
        if topmost {
            a.reverse();
        }
        a.into_iter().find(|d| markers.contains(d))
    }).collect();
    match roots.into_iter().collect::<Vec<_>>().as_slice() {
        [Some(r)] => Some(r.clone()),
        _ => None,
    }
}

// ---- containers ----

/// How a toolchain's containers run. The test run has no network, at most
/// `cpus` CPUs and `memory` of memory, and is killed after `timeout`
/// seconds (exit 137, a Timeout).
#[derive(Clone, Debug)]
pub struct Sandbox {
    pub image: String,
    pub cpus: String,
    pub memory: String,
    pub timeout: u64,
    /// Seconds the networked prefetch may take.
    pub prefetch_timeout: u64,
}

/// One `docker run`: `script` under `sh -c`, killed after `timeout`
/// seconds, as the owner of `tree` so every file it writes into a bind
/// mount stays removable by the miner.
pub struct Run<'a> {
    pub sandbox: &'a Sandbox,
    pub network: bool,
    pub tree: &'a Path,
    /// (host path or named volume, container path).
    pub mounts: Vec<(String, String)>,
    pub env: Vec<(String, String)>,
    pub workdir: String,
    pub script: String,
    pub timeout: u64,
}

/// Keep the last `n` characters of `s`.
pub fn tail(s: &str, n: usize) -> String {
    let count = s.chars().count();
    s.chars().skip(count.saturating_sub(n)).collect()
}

#[cfg(unix)]
fn owner(path: &Path) -> Result<(u32, u32)> {
    use std::os::unix::fs::MetadataExt;
    let m = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    Ok((m.uid(), m.gid()))
}

#[cfg(not(unix))]
fn owner(_: &Path) -> Result<(u32, u32)> {
    Ok((0, 0))
}

impl Run<'_> {
    /// The docker argv (without `docker`), for tests and the log.
    pub fn argv(&self) -> Result<Vec<String>> {
        let (uid, gid) = owner(self.tree)?;
        let mut a: Vec<String> = ["run", "--rm", "--init"].map(String::from).to_vec();
        if !self.network {
            a.extend(["--network", "none"].map(String::from));
        }
        a.extend(["--cpus".into(), self.sandbox.cpus.clone(), "--memory".into(), self.sandbox.memory.clone(),
                  "--user".into(), format!("{uid}:{gid}")]);
        for (k, v) in &self.env {
            a.extend(["-e".into(), format!("{k}={v}")]);
        }
        for (src, dst) in &self.mounts {
            a.extend(["-v".into(), format!("{src}:{dst}")]);
        }
        a.extend(["-w".into(), self.workdir.clone(), "--entrypoint".into(), "timeout".into(),
                  self.sandbox.image.clone(), "-s".into(), "KILL".into(), self.timeout.to_string(),
                  "sh".into(), "-c".into(), self.script.clone()]);
        Ok(a)
    }

    /// Run it; (exit code, combined output, last 20,000 characters of each).
    pub fn exec(&self) -> Result<(i32, String)> {
        let out = Command::new("docker").args(self.argv()?).output().context("running docker")?;
        let code = out.status.code().unwrap_or(-1);
        Ok((code, format!("{}{}", tail(&String::from_utf8_lossy(&out.stdout), 20_000),
                          tail(&String::from_utf8_lossy(&out.stderr), 20_000))))
    }
}

/// Make the top of named volume `volume` owned by `uid:gid` (named volumes
/// start root-owned; the toolchain containers run as the tree's owner).
pub fn own_volume(image: &str, volume: &str, tree: &Path) -> Result<()> {
    static DONE: std::sync::Mutex<BTreeSet<String>> = std::sync::Mutex::new(BTreeSet::new());
    let (uid, gid) = owner(tree)?;
    let key = format!("{volume} {uid}:{gid}");
    let mut done = DONE.lock().unwrap_or_else(|p| p.into_inner());
    if done.contains(&key) {
        return Ok(());
    }
    let out = Command::new("docker")
        .args(["run", "--rm", "--user", "0", "--network", "none", "--entrypoint", "chown",
               "-v", &format!("{volume}:/v"), image, &format!("{uid}:{gid}"), "/v"])
        .output().context("running docker")?;
    if !out.status.success() {
        bail!("chown of volume {volume}: {}", String::from_utf8_lossy(&out.stderr));
    }
    done.insert(key);
    Ok(())
}

/// Paths of the files under `root`, relative to it (hidden and ignored
/// files skipped): a checkout's own files, for telling the project's
/// missing files from a third party's.
pub fn tracked_like(root: &Path) -> BTreeSet<String> {
    ignore::WalkBuilder::new(root).build().flatten()
        .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
        .filter_map(|e| e.path().strip_prefix(root).ok().map(|p| p.to_string_lossy().replace('\\', "/")))
        .collect()
}

/// Single-quote `s` for `sh`.
pub fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// The container working directory of a project root in a tree at /src.
pub fn src_dir(root: &str) -> String {
    crate::workdir(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_split_into_roots_and_names() {
        assert_eq!(ancestors("a/b"), ["a/b", "a", ""]);
        assert_eq!(ancestors(""), [""]);
        assert!(under("", "x/y.py") && under("a", "a/y.py") && !under("a", "ab/y.py"));
        assert_eq!(rel("a", "a/b/c.py"), "b/c.py");
        assert_eq!(rel("", "b/c.py"), "b/c.py");
        assert!(in_dir_named("pkg/tests/x.py", &["tests"]) && !in_dir_named("tests.py", &["tests"]));
        assert_eq!((file_name("a/b/c.tar.gz"), extension("a/b/c.tar.gz")), ("c.tar.gz", "gz"));
        assert_eq!(extension("Makefile"), "");
        assert_eq!(sh_quote("it's"), "'it'\\''s'");
        assert_eq!(tail("abcdef", 3), "def");
    }

    #[test]
    fn a_single_root_is_the_nearest_or_topmost_marker() {
        let tracked: BTreeSet<String> = ["CMakeLists.txt", "lib/CMakeLists.txt", "lib/x.cc", "tools/p/pyproject.toml"]
            .map(String::from).into_iter().collect();
        let cm = |f: &str| file_name(f) == "CMakeLists.txt";
        let files = vec!["lib/x.cc".to_string()];
        assert_eq!(single_root(&files, &tracked, cm, false).as_deref(), Some("lib"));
        assert_eq!(single_root(&files, &tracked, cm, true).as_deref(), Some(""));
        let py = |f: &str| file_name(f) == "pyproject.toml";
        assert_eq!(single_root(&["tools/p/t/test_a.py".into()], &tracked, py, false).as_deref(), Some("tools/p"));
        assert_eq!(single_root(&["other/test_a.py".into()], &tracked, py, false), None, "no marker");
        assert_eq!(single_root(&["tools/p/a.py".into(), "other/b.py".into()], &tracked, py, false), None,
                   "two roots");
    }

    #[test]
    fn the_container_is_offline_capped_and_runs_as_the_tree_owner() {
        let d = tempfile::tempdir().unwrap();
        let sb = Sandbox { image: "img@sha256:00".into(), cpus: "4".into(), memory: "6g".into(),
                           timeout: 600, prefetch_timeout: 1800 };
        let run = Run { sandbox: &sb, network: false, tree: d.path(), mounts: vec![("vol".into(), "/c".into())],
                        env: vec![("HOME".into(), "/c".into())], workdir: "/src".into(),
                        script: "echo hi".into(), timeout: 600 };
        let a = run.argv().unwrap().join(" ");
        assert!(a.contains("--network none --cpus 4 --memory 6g --user "), "{a}");
        assert!(a.ends_with("-w /src --entrypoint timeout img@sha256:00 -s KILL 600 sh -c echo hi"), "{a}");
        let online = Run { network: true, ..run };
        assert!(!online.argv().unwrap().join(" ").contains("--network"));
    }
}
