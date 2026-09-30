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

//! Mine SWE-style Go tasks from a repository's own history.
//!
//! A task is one commit that changes Go source and the tests of the same
//! package. It is valid only if the commit's tests FAIL on the parent tree and
//! PASS on the commit tree (SWE-bench's FAIL_TO_PASS check), both run in a
//! network-less `golang` container with shared module and build caches.
//!
//! The `rrsi-mine` binary (src/main.rs) and the `rrsi_mine` Python module
//! (src/python.rs, feature `python`) are thin fronts over this library:
//!
//! ```text
//! rrsi-mine list --repo PATH [--since 2026-06-01]
//! rrsi-mine mine --repo PATH --out DIR [--since ...] [--jobs 4] [--limit N]
//! ```
//!
//! `mine` writes DIR/<sha12>/{task.json, src.patch, tests.patch, parent.log,
//! commit.log} and DIR/index.jsonl. It is resume-safe: a candidate with a
//! task.json is kept. The output holds the repository's code: keep DIR out of
//! any public repository.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
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
    pub detail: Option<String>,
    pub seconds: f64,
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
    /// container. Returns whether it passed and a log of the run.
    pub fn go_test(&self, tree: &Path, module_root: &str, packages: &[String]) -> Result<(bool, String)> {
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
        Ok((out.status.success(), log))
    }

    /// Fill the module cache once per module root, with network, before the
    /// network-less test runs.
    pub fn warm(&self, repo: &Path, sha: &str, root: &str) -> Result<()> {
        let dir = tempfile::Builder::new().prefix("rrsi-warm-").tempdir()?;
        export_tree(repo, sha, dir.path())?;
        let st = Command::new("docker")
            .args(["run", "--rm", "-e", "GOTOOLCHAIN=local",
                   "-v", &format!("{}:/go/pkg/mod", self.modcache),
                   "-v", &format!("{}:/src", dir.path().display()),
                   "-w", &workdir(root), self.image, "go", "mod", "download"])
            .stdout(Stdio::null()).stderr(Stdio::null()).status()?;
        if !st.success() {
            eprintln!("[mine] WARNING go mod download failed for {root}");
        }
        Ok(())
    }
}

pub fn apply_patch(tree: &Path, patch: &str) -> Result<Option<String>> {
    let mut child = Command::new("git").args(["apply", "-"]).current_dir(tree)
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn()?;
    child.stdin.take().context("git apply stdin")?.write_all(patch.as_bytes())?;
    let out = child.wait_with_output()?;
    Ok(if out.status.success() { None } else { Some(String::from_utf8_lossy(&out.stderr).into_owned()) })
}

pub fn validate(repo: &Path, out: &Path, cand: &Candidate, docker: &Docker) -> Result<TaskRecord> {
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
                               fails_before: None, passes_after: None, detail: None, seconds: 0.0 };
    if let Some(err) = apply_patch(&parent, &tests_patch)? {
        rec.reason = "tests.patch does not apply to parent".into();
        rec.detail = Some(err.chars().take(400).collect());
    } else {
        let (parent_ok, parent_log) = docker.go_test(&parent, &cand.module_root, &cand.packages)?;
        std::fs::write(tdir.join("parent.log"), parent_log)?;
        let (passes_after, commit_log) = docker.go_test(&commit, &cand.module_root, &cand.packages)?;
        std::fs::write(tdir.join("commit.log"), commit_log)?;
        let fails_before = !parent_ok;
        rec.valid = fails_before && passes_after;
        rec.reason = match (fails_before, passes_after) {
            (true, true) => "ok",
            (false, _) => "tests already pass on parent",
            (true, false) => "tests fail on the commit",
        }.into();
        rec.fails_before = Some(fails_before);
        rec.passes_after = Some(passes_after);
    }
    rec.seconds = (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0;
    std::fs::write(&task_json, serde_json::to_string_pretty(&rec)?)?;
    Ok(rec)
}

pub fn mine(repo: &Path, out: &Path, cands: Vec<Candidate>, jobs: usize, docker: &Docker) -> Result<()> {
    std::fs::create_dir_all(out)?;
    println!("[mine] {} candidates", cands.len());
    let roots: BTreeSet<&str> = cands.iter().map(|c| c.module_root.as_str()).collect();
    for root in roots {
        let newest = cands.iter().find(|c| c.module_root == root).expect("root has a candidate");
        println!("[mine] warming module cache for {} @ {}", if root.is_empty() { "." } else { root }, &newest.sha[..12]);
        docker.warm(repo, &newest.sha, root)?;
    }
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<TaskRecord>>> = Mutex::new((0..cands.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some(c) = cands.get(i) else { break };
                let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                match validate(repo, out, c, docker) {
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
        writeln!(index, "{}", serde_json::json!({
            "sha": r.cand.sha, "valid": r.valid, "reason": r.reason,
            "module_root": r.cand.module_root, "packages": r.cand.packages,
            "src_churn": r.cand.src_churn, "subject": r.cand.subject}))?;
    }
    println!("[mine] valid {valid}/{}", results.iter().flatten().count());
    Ok(())
}

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests {
    use super::*;

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
