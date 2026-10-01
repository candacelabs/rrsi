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

//! Mine SWE-style tasks from a repository's own history.
//!
//! A task is one commit that changes source and the tests of that source.
//! It is valid only if the commit's tests FAIL on the parent tree and PASS on
//! the commit tree (SWE-bench's FAIL_TO_PASS check), both run in a
//! network-less container after a networked step that prefetches the
//! dependencies. What "source", "tests" and "a run" mean is a toolchain's
//! business (src/toolchain): Go (`go test` in a `golang` container, the
//! default), Python (pytest), C++ (CMake + CTest) and Bazel (`bazel test`).
//!
//! The `rrsi-mine` binary (src/main.rs) and the `rrsi_mine` Python module
//! (src/python.rs, feature `python`) are thin fronts over this library:
//!
//! ```text
//! rrsi-mine list --repo PATH [--since 2026-06-01] [--toolchain go|python|cpp|bazel|auto]
//! rrsi-mine mine --repo PATH --out DIR [--since ...] [--jobs 4] [--limit N] [--toolchain ...]
//! ```
//!
//! `mine` writes DIR/<sha12>/{task.json, src.patch, tests.patch, parent.log,
//! commit.log} and DIR/index.jsonl. It is resume-safe: a candidate with a
//! task.json is kept. The output holds the repository's code: keep DIR out of
//! any public repository.
//!
//! The fairness stages (src/fairness.rs) then decide which valid tasks are
//! fair exam questions and write DIR/<sha12>/fairness/<stage>.json,
//! DIR/<sha12>/instruction.md and DIR/exam.jsonl:
//!
//! ```text
//! rrsi-mine fairness --tasks DIR --repo PATH [--jobs 2] [--only SHA12] [--force]
//! rrsi-mine flake|api|describe|probe|specificity|gate --tasks DIR ...
//! ```

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub const MAX_SRC_CHURN: u64 = 400;
pub const MAX_SRC_PACKAGES: usize = 3;
pub const GENERATED: [&str; 5] = ["/gen/", ".pb.go", "_cgen", "zz_generated", "_string.go"];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub sha: String,
    pub parent: String,
    pub subject: String,
    pub body: String,
    pub module_root: String,
    pub packages: Vec<String>,
    pub src_files: Vec<String>,
    pub test_files: Vec<String>,
    pub src_churn: u64,
    /// The toolchain that validates this candidate (`python`, `cpp`,
    /// `bazel`); absent means Go, so every earlier task.json still reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toolchain: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TaskRecord {
    #[serde(flatten)]
    pub cand: Candidate,
    pub valid: bool,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fails_before: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passes_after: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_outcome: Option<Outcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_outcome: Option<Outcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub seconds: f64,
}

/// What one `go test` run proved. Only `TestFail` and `BuildFail` are
/// evidence that the tests fail; `Infra` says nothing about the code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Pass,
    /// A test ran and failed.
    TestFail,
    /// The package or its tests did not compile (for a parent tree this is
    /// the normal case: the new tests call code that does not exist yet).
    BuildFail,
    /// The run never reached the code: module download, network, go.mod
    /// resolution or setup. Never counts as the tests failing.
    Infra,
    /// Killed by the timeout: ambiguous, never evidence either way.
    Timeout,
}

/// Log fragments that mean the toolchain could not assemble the build
/// inputs. Any of them makes the run `Infra`, whatever the exit code says.
pub const INFRA_MARKERS: [&str; 8] = [
    "module lookup disabled by GOPROXY",
    "missing go.sum entry",
    "go: updates to go.mod needed",
    "dial tcp",
    "i/o timeout",
    "no such host",
    "cannot find module providing package",
    "go: downloading",
];

/// The `module` path declared by the go.mod in `dir`, if readable.
pub fn module_path(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("go.mod")).ok()?.lines()
        .find_map(|l| l.strip_prefix("module ").map(|m| m.trim().trim_matches('"').to_string()))
}

/// Whether a toolchain line blames a package of the module under test
/// itself. Go reports a package the new tests import but the parent tree
/// lacks as "cannot find module providing package <own>/...: module lookup
/// disabled", word for word like a download failure; it is a build failure
/// of the code, not infrastructure.
fn blames_own_module(line: &str, own: Option<&str>) -> bool {
    match own {
        Some(m) => line.contains(&format!("providing package {m}/"))
            || line.contains(&format!("providing package {m}:")),
        None => false,
    }
}

/// Classify one `go test` run from its exit code, combined output and the
/// module path of the tree under test.
pub fn classify(exit: i32, log: &str, own_module: Option<&str>) -> Outcome {
    if exit == 0 {
        return Outcome::Pass;
    }
    if exit == 137 || exit == 124 {
        return Outcome::Timeout;
    }
    // "go: downloading" alone is only a download; with a failure it means
    // the cache was cold and the offline run could not finish assembling.
    let mut own_missing = false;
    for line in log.lines() {
        if blames_own_module(line, own_module) {
            own_missing = true;
        } else if INFRA_MARKERS.iter().any(|m| line.contains(m)) {
            return Outcome::Infra;
        }
    }
    if own_missing || log.contains("[build failed]") || log.contains("undefined: ")
        || log.contains("[setup failed]") && log.contains(".go:") && !log.contains("module") {
        return Outcome::BuildFail;
    }
    if log.contains("--- FAIL") || log.contains("[FAIL]") || log.contains("FAIL\t") {
        return Outcome::TestFail;
    }
    Outcome::Infra
}

/// FAIL_TO_PASS: valid only when the parent run really failed (a test
/// failed or the new tests did not compile) and the commit run passed.
/// An `Infra` or `Timeout` run on either side is never evidence.
pub fn decide(parent: Outcome, commit: Outcome) -> (bool, &'static str) {
    use Outcome::*;
    match (parent, commit) {
        (Infra, _) | (_, Infra) => (false, "infra: a run could not assemble its build inputs"),
        (Timeout, _) | (_, Timeout) => (false, "timeout: a run was killed"),
        (Pass, _) => (false, "tests already pass on parent"),
        (_, TestFail) | (_, BuildFail) => (false, "tests fail on the commit"),
        (TestFail, Pass) | (BuildFail, Pass) => (true, "ok"),
    }
}

pub struct Docker<'a> {
    pub image: &'a str,
    pub modcache: &'a str,
    pub buildcache: &'a str,
    pub test_timeout: u64,
}

pub fn git(repo: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git").args(args).current_dir(repo).output()
        .with_context(|| format!("running git {args:?}"))?;
    if !out.status.success() {
        bail!("git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn git_ok(repo: &Path, args: &[&str]) -> bool {
    Command::new("git").args(args).current_dir(repo)
        .stdout(Stdio::null()).stderr(Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false)
}

pub fn dir_of(path: &str) -> String {
    path.rsplit_once('/').map(|(d, _)| d.to_string()).unwrap_or_default()
}

/// Nearest directory at or above `dir` holding a go.mod at `sha`.
pub fn module_root(repo: &Path, sha: &str, dir: &str) -> Option<String> {
    let parts: Vec<&str> = if dir.is_empty() { vec![] } else { dir.split('/').collect() };
    (0..=parts.len()).rev().map(|i| parts[..i].join("/")).find(|d| {
        let spec = if d.is_empty() { format!("{sha}:go.mod") } else { format!("{sha}:{d}/go.mod") };
        git_ok(repo, &["cat-file", "-e", &spec])
    })
}

pub fn relative_package(root: &str, pkg: &str) -> String {
    let rel = if root.is_empty() { pkg } else { pkg.strip_prefix(root).unwrap_or(pkg).trim_start_matches('/') };
    if rel.is_empty() { "./".to_string() } else { format!("./{rel}") }
}

pub fn candidates(repo: &Path, since: &str) -> Result<Vec<Candidate>> {
    let since_arg = format!("--since={since}");
    let log = git(repo, &["log", &since_arg, "--no-merges", "--format=%H", "--", "*_test.go"])?;
    let mut out = Vec::new();
    for sha in log.split_whitespace() {
        let mut src: Vec<(u64, String)> = Vec::new();
        let mut tests: Vec<String> = Vec::new();
        for line in git(repo, &["show", "--numstat", "--format=", sha])?.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 3 || !f[2].ends_with(".go") {
                continue;
            }
            let churn = f[0].parse::<u64>().unwrap_or(0) + f[1].parse::<u64>().unwrap_or(0);
            if f[2].ends_with("_test.go") {
                tests.push(f[2].to_string());
            } else if !GENERATED.iter().any(|g| f[2].contains(g)) {
                src.push((churn, f[2].to_string()));
            }
        }
        let pk_src: BTreeSet<String> = src.iter().map(|(_, f)| dir_of(f)).collect();
        let pk_tst: BTreeSet<String> = tests.iter().map(|f| dir_of(f)).collect();
        let shared: Vec<String> = pk_src.intersection(&pk_tst).cloned().collect();
        let churn: u64 = src.iter().map(|(c, _)| c).sum();
        if src.is_empty() || shared.is_empty() || churn > MAX_SRC_CHURN || pk_src.len() > MAX_SRC_PACKAGES {
            continue;
        }
        let roots: BTreeSet<Option<String>> = shared.iter().map(|p| module_root(repo, sha, p)).collect();
        let root = match roots.into_iter().collect::<Vec<_>>().as_slice() {
            [Some(r)] => r.clone(),
            _ => continue,
        };
        out.push(Candidate {
            sha: sha.to_string(),
            parent: git(repo, &["rev-parse", &format!("{sha}^")])?.trim().to_string(),
            subject: git(repo, &["log", "-1", "--format=%s", sha])?.trim().to_string(),
            body: git(repo, &["log", "-1", "--format=%b", sha])?.trim().to_string(),
            packages: shared.iter().map(|p| relative_package(&root, p)).collect(),
            module_root: root,
            src_files: src.into_iter().map(|(_, f)| f).collect(),
            test_files: tests,
            src_churn: churn,
            toolchain: None,
        });
    }
    Ok(out)
}

pub fn export_tree(repo: &Path, sha: &str, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    let mut archive = Command::new("git").args(["archive", "--format=tar", sha])
        .current_dir(repo).stdout(Stdio::piped()).spawn()?;
    let tar = Command::new("tar").arg("-x").arg("-C").arg(dest)
        .stdin(archive.stdout.take().context("git archive stdout")?).status()?;
    if !archive.wait()?.success() || !tar.success() {
        bail!("exporting {sha} failed");
    }
    Ok(())
}

pub fn workdir(root: &str) -> String {
    if root.is_empty() { "/src".into() } else { format!("/src/{root}") }
}

impl Docker<'_> {
    /// `go test` of `packages` under `module_root` in a network-less
    /// container. Returns the classified outcome and a log of the run.
    pub fn go_test(&self, tree: &Path, module_root: &str, packages: &[String]) -> Result<(Outcome, String)> {
        let mount = format!("{}:/src", tree.display());
        let mut cmd = Command::new("docker");
        cmd.args(["run", "--rm", "--network", "none", "--cpus", "4", "--memory", "6g",
                  "-e", "GOFLAGS=-mod=mod", "-e", "GOTOOLCHAIN=local", "-e", "GOPROXY=off",
                  "-v", &format!("{}:/go/pkg/mod", self.modcache),
                  "-v", &format!("{}:/root/.cache/go-build", self.buildcache),
                  "-v", &mount, "-w", &workdir(module_root), self.image,
                  "timeout", "-s", "KILL", &self.test_timeout.to_string(),
                  "go", "test", "-count=1"]);
        cmd.args(packages);
        let out = cmd.output().context("running docker")?;
        let tail = |b: &[u8]| {
            let s = String::from_utf8_lossy(b);
            s.chars().rev().take(20_000).collect::<String>().chars().rev().collect::<String>()
        };
        let log = format!("$ go test -count=1 {}\nexit={}\n{}{}",
            packages.join(" "), out.status.code().unwrap_or(-1),
            tail(&out.stdout), tail(&out.stderr));
        let own = module_path(&tree.join(module_root));
        Ok((classify(out.status.code().unwrap_or(-1), &log, own.as_deref()), log))
    }

    /// Download every module `tree` needs into the shared cache, with
    /// network, so the network-less test run of THIS tree can build. A cache
    /// warmed at some other commit is not enough: older commits pin other
    /// module versions (2026-09-30: this made valid tasks look invalid, and
    /// could make a parent's download failure look like a failing test).
    pub fn download(&self, tree: &Path, module_root: &str) -> Result<(bool, String)> {
        let out = Command::new("docker")
            .args(["run", "--rm", "-e", "GOTOOLCHAIN=local", "-e", "GOFLAGS=-mod=mod",
                   "-v", &format!("{}:/go/pkg/mod", self.modcache),
                   "-v", &format!("{}:/src", tree.display()),
                   "-w", &workdir(module_root), self.image, "go", "mod", "download", "all"])
            .output().context("running docker")?;
        Ok((out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned()))
    }
}

pub fn apply_patch(tree: &Path, patch: &str) -> Result<Option<String>> {
    let mut child = Command::new("git").args(["apply", "-"]).current_dir(tree)
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn()?;
    child.stdin.take().context("git apply stdin")?.write_all(patch.as_bytes())?;
    let out = child.wait_with_output()?;
    Ok(if out.status.success() { None } else { Some(String::from_utf8_lossy(&out.stderr).into_owned()) })
}

pub fn validate(repo: &Path, out: &Path, cand: &Candidate, tc: &dyn Toolchain) -> Result<TaskRecord> {
    let tdir = out.join(&cand.sha[..12]);
    let task_json = tdir.join("task.json");
    if task_json.is_file() {
        return Ok(serde_json::from_str(&std::fs::read_to_string(&task_json)?)?);
    }
    std::fs::create_dir_all(&tdir)?;
    let t0 = Instant::now();
    let diff = |files: &[String]| -> Result<String> {
        let mut args = vec!["diff", cand.parent.as_str(), cand.sha.as_str(), "--"];
        args.extend(files.iter().map(String::as_str));
        git(repo, &args)
    };
    let tests_patch = diff(&cand.test_files)?;
    std::fs::write(tdir.join("tests.patch"), &tests_patch)?;
    std::fs::write(tdir.join("src.patch"), diff(&cand.src_files)?)?;

    let work = tempfile::Builder::new().prefix("rrsi-mine-").tempdir()?;
    let (parent, commit) = (work.path().join("parent"), work.path().join("commit"));
    export_tree(repo, &cand.parent, &parent)?;
    export_tree(repo, &cand.sha, &commit)?;
    let mut rec = TaskRecord { cand: cand.clone(), valid: false, reason: String::new(),
                               fails_before: None, passes_after: None, parent_outcome: None,
                               commit_outcome: None, detail: None, seconds: 0.0 };
    if let Some(err) = apply_patch(&parent, &tests_patch)? {
        rec.reason = "tests.patch does not apply to parent".into();
        rec.detail = Some(err.chars().take(400).collect());
    } else {
        let short = &cand.sha[..12];
        let mut infra = None;
        for (name, tree) in [("parent", &parent), ("commit", &commit)] {
            println!("[mine]   {short} {name}: {}", tc.prefetch_label());
            let (ok, log) = tc.prefetch(tree, cand)?;
            if !ok {
                infra = Some(format!("{name}: {} failed: {}", tc.prefetch_label(),
                                     log.chars().rev().take(300).collect::<String>().chars().rev().collect::<String>()));
            }
        }
        // The units the tests run as, resolved on the commit tree (a CTest
        // name, a Bazel label); Go and Python keep the listed ones.
        let units = tc.resolve_units(&commit, cand)?;
        if units.is_empty() {
            rec.reason = "no test unit runs the changed tests".into();
            rec.detail = infra;
        } else {
            if units != cand.packages {
                println!("[mine]   {short} units: {}", units.join(" "));
                rec.cand.packages = units;
            }
            let cand = &rec.cand.clone();
            let refetch = |name: &str, tree: &Path| -> Result<()> {
                if tc.prefetch_before_each_run() {
                    println!("[mine]   {short} {name}: {} (again, right before the run)", tc.prefetch_label());
                    tc.prefetch(tree, cand)?;
                }
                Ok(())
            };
            refetch("parent", &parent)?;
            println!("[mine]   {short} parent: {}", tc.test_label(&cand.packages));
            let (parent_out, parent_log) = tc.run_tests(&parent, cand)?;
            println!("[mine]   {short} parent: {parent_out:?}");
            std::fs::write(tdir.join("parent.log"), parent_log)?;
            refetch("commit", &commit)?;
            println!("[mine]   {short} commit: {}", tc.test_label(&cand.packages));
            let (commit_out, commit_log) = tc.run_tests(&commit, cand)?;
            println!("[mine]   {short} commit: {commit_out:?}");
            std::fs::write(tdir.join("commit.log"), commit_log)?;
            let (valid, reason) = decide(parent_out, commit_out);
            rec.valid = valid;
            rec.reason = reason.into();
            rec.parent_outcome = Some(parent_out);
            rec.commit_outcome = Some(commit_out);
            rec.fails_before = Some(matches!(parent_out, Outcome::TestFail | Outcome::BuildFail));
            rec.passes_after = Some(commit_out == Outcome::Pass);
            if infra.is_some() {
                rec.detail = infra;
            }
        }
    }
    rec.seconds = (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0;
    write_record(&tdir, &rec)?;
    Ok(rec)
}

/// Record a validation in `tdir`. An infra or timeout verdict is not a fact
/// about the commit: it goes to retry.json and no task.json is written, so
/// the next run retries it instead of reusing the verdict. A final verdict
/// goes to task.json and removes any retry.json an earlier attempt left, so
/// a task directory never holds both.
pub fn write_record(tdir: &Path, rec: &TaskRecord) -> Result<()> {
    let json = serde_json::to_string_pretty(rec)?;
    if matches!(rec.parent_outcome, Some(Outcome::Infra | Outcome::Timeout))
        || matches!(rec.commit_outcome, Some(Outcome::Infra | Outcome::Timeout)) {
        std::fs::write(tdir.join("retry.json"), json)?;
    } else {
        std::fs::write(tdir.join("task.json"), json)?;
        match std::fs::remove_file(tdir.join("retry.json")) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
            _ => {}
        }
    }
    Ok(())
}

/// The git work tree containing `path` (or its nearest existing ancestor),
/// if any. Mined tasks carry a repository's source: they must never be
/// written inside a work tree, where they could be committed or published.
pub fn enclosing_work_tree(path: &Path) -> Option<PathBuf> {
    let abs = if path.is_absolute() { path.to_path_buf() }
              else { std::env::current_dir().ok()?.join(path) };
    abs.ancestors().find(|a| a.join(".git").exists()).map(Path::to_path_buf)
}

pub fn mine(repo: &Path, out: &Path, cands: Vec<Candidate>, jobs: usize, tcs: &Toolchains) -> Result<()> {
    if let Some(tree) = enclosing_work_tree(out) {
        bail!("refusing to write mined tasks to {} inside the git work tree {}: \
               they contain repository source", out.display(), tree.display());
    }
    std::fs::create_dir_all(out)?;
    println!("[mine] {} candidates", cands.len());
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<TaskRecord>>> = Mutex::new((0..cands.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some(c) = cands.get(i) else { break };
                println!("[mine] start {} {}", &c.sha[..12], c.subject.chars().take(60).collect::<String>());
                let res = tcs.for_candidate(c).and_then(|tc| validate(repo, out, c, tc));
                let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                match res {
                    Ok(r) => {
                        println!("[mine] {n}/{} {} valid={} ({}, {}s) {}", cands.len(), &c.sha[..12],
                                 r.valid, r.reason, r.seconds, c.subject.chars().take(60).collect::<String>());
                        results.lock().expect("results lock")[i] = Some(r);
                    }
                    Err(e) => eprintln!("[mine] {n}/{} {} ERROR {e:#}", cands.len(), &c.sha[..12]),
                }
            });
        }
    });
    let results = results.into_inner().expect("results lock");
    let mut index = std::fs::File::create(out.join("index.jsonl"))?;
    let mut valid = 0;
    for r in results.iter().flatten() {
        valid += r.valid as usize;
        let mut line = serde_json::json!({
            "sha": r.cand.sha, "valid": r.valid, "reason": r.reason,
            "module_root": r.cand.module_root, "packages": r.cand.packages,
            "src_churn": r.cand.src_churn, "subject": r.cand.subject});
        if let Some(t) = &r.cand.toolchain {
            line["toolchain"] = t.clone().into();
        }
        writeln!(index, "{line}")?;
    }
    println!("[mine] valid {valid}/{}", results.iter().flatten().count());
    Ok(())
}

pub mod fairness;
pub mod history;
pub mod llm;
pub mod scan;
pub mod toolchain;

pub use toolchain::{Toolchain, Toolchains};

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests {
    use super::*;

    // Regression, 2026-09-30: the module cache was warmed only at the newest
    // commit per module, so offline runs of older trees failed with these
    // exact lines and were scored as "tests fail". A download failure must
    // never count as the tests failing, on either side.
    const REAL_INFRA_LOG: &str = "$ go test -count=1 ./services/candaceos/webui\nexit=1\n\
        FAIL\tgithub.com/candacelabs/candace/services/candaceos/webui [setup failed]\nFAIL\n\
        go: downloading github.com/gin-contrib/sse v1.1.0\n\
        # github.com/candacelabs/candace/services/candaceos/webui\n\
        /go/pkg/mod/github.com/gin-gonic/gin@v1.12.0/context.go:25:2: \
        module lookup disabled by GOPROXY=off\n";

    #[test]
    fn a_module_download_failure_is_infra_not_a_failing_test() {
        assert_eq!(classify(1, REAL_INFRA_LOG, Some("github.com/candacelabs/candace")), Outcome::Infra);
        assert_eq!(classify(1, "x.go:3:2: missing go.sum entry for module providing package y", None),
                   Outcome::Infra);
    }

    // Regression, 2026-09-30 (task 42fe7c3aa0b1): a new test importing a
    // package of the module under test that the parent lacks is reported as
    // "cannot find module providing package <own>/...: module lookup
    // disabled". That is the code failing to build, not infrastructure.
    const REAL_OWN_PACKAGE_LOG: &str = "FAIL\n\
        go: finding module for package github.com/candacelabs/csf/services/warden/internal/transportidentity\n\
        # github.com/candacelabs/csf/services/warden/election\n\
        services/warden/election/harness_test.go:15:2: cannot find module providing package \
        github.com/candacelabs/csf/services/warden/internal/transportidentity: module lookup disabled by GOPROXY=off\n";

    #[test]
    fn a_missing_package_of_the_module_itself_is_a_build_failure() {
        assert_eq!(classify(1, REAL_OWN_PACKAGE_LOG, Some("github.com/candacelabs/csf")),
                   Outcome::BuildFail);
        // The same line naming a THIRD-PARTY module is still infrastructure.
        assert_eq!(classify(1, REAL_OWN_PACKAGE_LOG, Some("github.com/other/module")), Outcome::Infra);
        assert_eq!(classify(1, REAL_OWN_PACKAGE_LOG, None), Outcome::Infra);
        // Own-module blame never hides a real download failure elsewhere.
        let both = format!("{REAL_OWN_PACKAGE_LOG}\n{REAL_INFRA_LOG}");
        assert_eq!(classify(1, &both, Some("github.com/candacelabs/csf")), Outcome::Infra);
    }

    #[test]
    fn module_path_reads_go_mod() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("go.mod"), "// c\nmodule github.com/candacelabs/csf\n\ngo 1.26\n").unwrap();
        assert_eq!(module_path(d.path()).as_deref(), Some("github.com/candacelabs/csf"));
        assert!(module_path(&d.path().join("missing")).is_none());
    }

    #[test]
    fn infra_on_either_side_never_makes_a_task_valid() {
        use Outcome::*;
        for other in [Pass, TestFail, BuildFail, Infra, Timeout] {
            assert!(!decide(Infra, other).0, "parent Infra, commit {other:?}");
            assert!(!decide(other, Infra).0, "parent {other:?}, commit Infra");
            assert!(!decide(Timeout, other).0 && !decide(other, Timeout).0);
        }
    }

    #[test]
    fn only_a_real_failure_then_a_pass_is_valid() {
        use Outcome::*;
        assert_eq!(decide(TestFail, Pass), (true, "ok"));
        assert_eq!(decide(BuildFail, Pass), (true, "ok"));
        assert!(!decide(Pass, Pass).0);
        assert!(!decide(TestFail, TestFail).0);
        assert!(!decide(BuildFail, BuildFail).0);
    }

    #[test]
    fn real_failures_classify_as_failures() {
        assert_eq!(classify(0, "ok  \tpkg\t0.1s", None), Outcome::Pass);
        assert_eq!(classify(1, "--- FAIL: TestX (0.00s)\nFAIL\tpkg\t0.1s", None), Outcome::TestFail);
        assert_eq!(classify(1, "Ran 3 of 3 Specs\n[FAIL] Mailbox rejects duplicates\nFAIL\tpkg", None),
                   Outcome::TestFail);
        assert_eq!(classify(1, "# pkg [pkg.test]\n./a_test.go:9:5: undefined: NewThing\n\
                               FAIL\tpkg [build failed]", None), Outcome::BuildFail);
        assert_eq!(classify(137, "", None), Outcome::Timeout);
        // An exit with nothing recognizable is not evidence of a failing test.
        assert_eq!(classify(1, "something odd", None), Outcome::Infra);
    }

    #[test]
    fn task_output_inside_any_work_tree_is_refused() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir(d.path().join(".git")).unwrap();
        let inside = d.path().join("a/b/tasks");
        assert_eq!(enclosing_work_tree(&inside).as_deref(), Some(d.path()));
        let outside = tempfile::tempdir().unwrap();
        assert!(enclosing_work_tree(&outside.path().join("tasks")).is_none());
        let tcs = Toolchains { all: vec![] };
        let err = mine(d.path(), &inside, vec![], 1, &tcs).unwrap_err();
        assert!(format!("{err}").contains("inside the git work tree"));
        assert!(!inside.exists(), "nothing may be created inside the work tree");
    }

    // Regression, 2026-09-30: a task retried after an infra run kept its
    // stale retry.json next to the new task.json.
    #[test]
    fn a_final_verdict_removes_the_earlier_retry_record() {
        let d = tempfile::tempdir().unwrap();
        let cand = Candidate { sha: "a".repeat(40), parent: "b".repeat(40), subject: "s".into(),
                               body: String::new(), module_root: "m".into(), packages: vec![],
                               src_files: vec![], test_files: vec![], src_churn: 1, toolchain: None };
        let rec = |parent, commit| TaskRecord {
            cand: cand.clone(), valid: false, reason: String::new(), fails_before: None,
            passes_after: None, parent_outcome: Some(parent), commit_outcome: Some(commit),
            detail: None, seconds: 0.0 };
        write_record(d.path(), &rec(Outcome::BuildFail, Outcome::Infra)).unwrap();
        assert!(d.path().join("retry.json").is_file() && !d.path().join("task.json").exists());
        write_record(d.path(), &rec(Outcome::BuildFail, Outcome::Pass)).unwrap();
        assert!(d.path().join("task.json").is_file());
        assert!(!d.path().join("retry.json").exists(), "never both files");
        // A first-time final verdict with no retry.json is fine too.
        let fresh = tempfile::tempdir().unwrap();
        write_record(fresh.path(), &rec(Outcome::TestFail, Outcome::Pass)).unwrap();
        assert!(fresh.path().join("task.json").is_file());
    }

    #[test]
    fn task_json_without_a_toolchain_is_a_go_task_and_go_writes_none() {
        let old = r#"{"sha":"a","parent":"b","subject":"s","body":"","module_root":"m","packages":["./p"],
                      "src_files":[],"test_files":[],"src_churn":1,"valid":true,"reason":"ok","seconds":1.0}"#;
        let rec: TaskRecord = serde_json::from_str(old).unwrap();
        assert_eq!(rec.cand.toolchain, None);
        assert!(!serde_json::to_string(&rec).unwrap().contains("toolchain"), "Go output is unchanged");
        let mut py = rec.cand.clone();
        py.toolchain = Some("python".into());
        assert!(serde_json::to_string(&py).unwrap().contains(r#""toolchain":"python""#));
    }

    #[test]
    fn packages_are_relative_to_the_module_root() {
        assert_eq!(relative_package("candace", "candace/services/email"), "./services/email");
        assert_eq!(relative_package("candace", "candace"), "./");
        assert_eq!(relative_package("", "pkg/x"), "./pkg/x");
        assert_eq!(dir_of("a/b/c.go"), "a/b");
        assert_eq!(dir_of("c.go"), "");
        assert_eq!(workdir(""), "/src");
    }

    #[test]
    fn candidates_and_module_roots_from_a_scratch_repo() {
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        let run = |args: &[&str]| assert!(Command::new("git").args(args).current_dir(r)
            .env("GIT_AUTHOR_NAME", "t").env("GIT_AUTHOR_EMAIL", "t@example.invalid")
            .env("GIT_COMMITTER_NAME", "t").env("GIT_COMMITTER_EMAIL", "t@example.invalid")
            .status().unwrap().success());
        run(&["init", "-q"]);
        std::fs::create_dir_all(r.join("mod/pkg")).unwrap();
        std::fs::write(r.join("mod/go.mod"), "module example.invalid/m\n").unwrap();
        std::fs::write(r.join("mod/pkg/a.go"), "package pkg\n").unwrap();
        run(&["add", "-A"]);
        run(&["commit", "-qm", "base"]);
        std::fs::write(r.join("mod/pkg/a.go"), "package pkg\n\nfunc A() int { return 1 }\n").unwrap();
        std::fs::write(r.join("mod/pkg/a_test.go"), "package pkg\n").unwrap();
        run(&["add", "-A"]);
        run(&["commit", "-qm", "feat: add A"]);
        let c = candidates(r, "2000-01-01").unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].module_root, "mod");
        assert_eq!(c[0].packages, vec!["./pkg".to_string()]);
        assert_eq!(c[0].subject, "feat: add A");
        assert_eq!(c[0].src_files, vec!["mod/pkg/a.go".to_string()]);
    }
}
