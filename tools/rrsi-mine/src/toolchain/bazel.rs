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

//! Bazel: language-agnostic `bazel test` of the test targets built from the
//! changed test files.
//!
//! The workspace root is the nearest directory holding `MODULE.bazel`,
//! `WORKSPACE` or `WORKSPACE.bazel`. BUILD files belong to the tests: a
//! commit's new test target is wired in its BUILD file, and a BUILD file
//! naming sources the parent lacks fails the parent's build, which is the
//! evidence FAIL_TO_PASS needs. A candidate's units start as `//pkg:all` for
//! each package whose BUILD file names a changed test file; on the commit
//! tree they narrow to `tests(rdeps(//pkg:all, <changed files>))` minus
//! `manual` targets.
//!
//! Every container shares one output base, repository cache and disk cache
//! in a named volume, so the networked prefetch (`bazel test --nobuild
//! --keep_going`, which fetches every external repository the targets need)
//! leaves the offline run nothing to download. One output base admits one
//! Bazel server at a time, so Bazel containers run one after another.

use super::{sh_quote, Changed, FileKind, Mapper, Run, Sandbox, Toolchain};
use crate::{dir_of, git, Candidate, Outcome, GENERATED};
use anyhow::Result;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Mutex;

pub struct Bazel {
    pub sandbox: Sandbox,
    /// Named volume: output base, repository cache, disk cache and HOME.
    pub cache: String,
    /// One Bazel server per output base.
    pub lock: Mutex<()>,
}

pub const BUILD_FILES: [&str; 2] = ["BUILD", "BUILD.bazel"];
pub const TEST_DIRS: [&str; 5] = ["test", "tests", "testdata", "testing", "__tests__"];

/// A test file by the conventions of the languages Bazel builds here.
pub fn is_test_name(path: &str) -> bool {
    let n = super::file_name(path);
    let stem = n.rsplit_once('.').map(|(s, _)| s).unwrap_or(n);
    stem.ends_with("_test") || stem.ends_with("-test") || stem.ends_with("Test")
        || stem.ends_with("_spec") || (n.ends_with(".py") && stem.starts_with("test_"))
}

pub fn classify_file(path: &str) -> FileKind {
    let n = super::file_name(path);
    let ext = super::extension(path);
    if BUILD_FILES.contains(&n) {
        return FileKind::Test;
    }
    if matches!(n, "MODULE.bazel" | "MODULE.bazel.lock" | "WORKSPACE" | "WORKSPACE.bazel") || ext == "bzl" {
        return FileKind::Source;
    }
    if GENERATED.iter().any(|g| path.contains(g)) || n.ends_with("_pb2.py") || n.contains(".pb.") {
        return FileKind::Generated;
    }
    if is_test_name(path) || super::in_dir_named(path, &TEST_DIRS) {
        return FileKind::Test;
    }
    if matches!(ext, "md" | "txt" | "rst" | "png" | "svg" | "jpg" | "gif" | "pdf" | "html" | "css")
        || n.starts_with('.') || n == "LICENSE" {
        return FileKind::Other;
    }
    FileKind::Source
}

pub fn is_root_marker(path: &str) -> bool {
    matches!(super::file_name(path), "MODULE.bazel" | "WORKSPACE" | "WORKSPACE.bazel")
}

/// The Bazel package directory of `file` (nearest ancestor holding a BUILD
/// file, within `root`), relative to the repository.
pub fn package_of(file: &str, root: &str, tracked: &BTreeSet<String>) -> Option<String> {
    super::ancestors(&dir_of(file)).into_iter()
        .take_while(|d| super::under(root, d) || d == root)
        .find(|d| BUILD_FILES.iter().any(|b| tracked.contains(&join(d, b))))
}

fn join(dir: &str, name: &str) -> String {
    if dir.is_empty() { name.to_string() } else { format!("{dir}/{name}") }
}

/// The label of a repository path in the workspace at `root`.
pub fn label(root: &str, pkg: &str, file: Option<&str>) -> String {
    let p = super::rel(root, pkg);
    let p = if pkg == root { "" } else { p };
    match file {
        Some(f) => format!("//{p}:{}", super::rel(pkg, f)),
        None => format!("//{p}:all"),
    }
}

struct BazelMapper;

impl Mapper for BazelMapper {
    fn root(&self, ch: &Changed) -> Option<String> {
        let tests: Vec<String> = ch.tests.iter().filter(|f| !BUILD_FILES.contains(&super::file_name(f)))
            .cloned().collect();
        super::single_root(if tests.is_empty() { &ch.tests } else { &tests }, &ch.tracked, is_root_marker, false)
    }

    /// `//pkg:all` for every package whose BUILD file declares a test rule
    /// and names (or globs) one of the changed test files.
    fn units(&self, ch: &Changed, root: &str, repo: &Path) -> Result<Vec<String>> {
        let mut out = BTreeSet::new();
        for f in ch.tests.iter().filter(|f| !BUILD_FILES.contains(&super::file_name(f)) && ch.tracked.contains(*f)) {
            let Some(pkg) = package_of(f, root, &ch.tracked) else { continue };
            let build = BUILD_FILES.iter().map(|b| join(&pkg, b)).find(|b| ch.tracked.contains(b));
            let Some(build) = build else { continue };
            let text = git(repo, &["show", &format!("{}:{build}", ch.sha)])?;
            let named = text.contains(&format!("\"{}\"", super::rel(&pkg, f))) || text.contains("glob(");
            if named && text.contains("_test(") {
                out.insert(label(root, &pkg, None));
            }
        }
        Ok(out.into_iter().collect())
    }
}

const STARTUP: &str = "--output_user_root=/cache/out";
const COMMON: &str = "--repository_cache=/cache/repos --disk_cache=/cache/disk \
    --experimental_convenience_symlinks=ignore --color=no --curses=no --noshow_progress";

pub fn prefetch_script(units: &[String]) -> String {
    format!("bazel {STARTUP} test --nobuild --keep_going --build_tests_only {COMMON} {}\n", quoted(units))
}

pub fn query_script(pattern: &str, files: &[String]) -> String {
    let set = files.join(" ");
    let q = format!("tests(rdeps({pattern}, set({set}))) except attr(tags, '\\bmanual\\b', {pattern})");
    format!("bazel {STARTUP} query --keep_going --output=label {COMMON_QUERY} {}\n", sh_quote(&q))
}

const COMMON_QUERY: &str = "--repository_cache=/cache/repos --color=no --curses=no --noshow_progress";

pub fn test_script(units: &[String]) -> String {
    format!("bazel {STARTUP} test --build_tests_only --test_output=errors {COMMON} {}\n", quoted(units))
}

fn quoted(units: &[String]) -> String {
    units.iter().map(|u| sh_quote(u)).collect::<Vec<_>>().join(" ")
}

/// Lines that mean the run never reached the code: downloads, registries,
/// repositories the offline run does not have.
pub const INFRA_MARKERS: [&str; 14] = [
    "Error downloading",
    "UnknownHostException",
    "Unknown host",
    "Failed to fetch registry file",
    "Error computing the main repository mapping",
    "No repository visible as",
    "fetching repository",
    "Network is unreachable",
    "No space left on device",
    "Server terminated abruptly",
    "OutOfMemoryError",
    "No test targets were found",
    "Couldn't download",
    "error loading package '@",
];

/// Whether a line blames a package, target or file of the main repository
/// (a label without a repository, or `@@//`): the parent lacks code the fix
/// adds. `no such package '@foo...` is an external repository: not ours.
pub fn blames_own(line: &str) -> bool {
    for marker in ["no such package '", "no such target '", "missing input file '"] {
        if let Some(i) = line.find(marker) {
            let rest = &line[i + marker.len()..];
            return !rest.starts_with('@') || rest.starts_with("@@//") || rest.starts_with("@//");
        }
    }
    false
}

/// Classify one `bazel test` run (exit 3: tests failed; 1: build failed;
/// 4: nothing to test).
pub fn classify(exit: i32, log: &str) -> Outcome {
    if exit == 0 {
        return Outcome::Pass;
    }
    if exit == 137 || exit == 124 {
        return Outcome::Timeout;
    }
    let mut own_missing = false;
    for line in log.lines() {
        if blames_own(line) {
            own_missing = true;
            continue;
        }
        if INFRA_MARKERS.iter().any(|m| line.contains(m)) {
            return Outcome::Infra;
        }
    }
    if own_missing {
        return Outcome::BuildFail;
    }
    if exit == 3 || log.contains(" FAILED in ") || log.contains("fails locally") {
        return Outcome::TestFail;
    }
    if exit == 1 && (log.contains("Build did NOT complete successfully") || log.contains(" failed: ")) {
        return Outcome::BuildFail;
    }
    Outcome::Infra
}

impl Bazel {
    fn run<'a>(&'a self, tree: &'a Path, cand: &Candidate, network: bool, script: String) -> Run<'a> {
        Run { sandbox: &self.sandbox, network, tree,
              mounts: vec![(tree.display().to_string(), "/src".into()), (self.cache.clone(), "/cache".into())],
              env: vec![("HOME".into(), "/cache/home".into())],
              workdir: super::src_dir(&cand.module_root), script,
              timeout: if network { self.sandbox.prefetch_timeout } else { self.sandbox.timeout } }
    }

    fn exec(&self, run: Run<'_>) -> Result<(i32, String)> {
        let _one = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        run.exec()
    }
}

impl Toolchain for Bazel {
    fn name(&self) -> &'static str {
        "bazel"
    }

    fn is_root_marker(&self, path: &str) -> bool {
        is_root_marker(path)
    }

    fn classify_file(&self, path: &str) -> FileKind {
        classify_file(path)
    }

    fn candidates(&self, repo: &Path, since: &str) -> Result<Vec<Candidate>> {
        super::scan_candidates(self, &BazelMapper, repo, since)
    }

    /// Narrow each `//pkg:all` to the non-manual test targets depending on
    /// the changed files of that package; keep `//pkg:all` when the query
    /// fails or finds nothing.
    fn resolve_units(&self, commit_tree: &Path, cand: &Candidate) -> Result<Vec<String>> {
        let root = &cand.module_root;
        let tracked = super::tracked_like(commit_tree);
        let mut out = BTreeSet::new();
        for unit in &cand.packages {
            let pkg_rel = unit.trim_start_matches("//").trim_end_matches(":all");
            let pkg = if pkg_rel.is_empty() { root.clone() } else { join(root, pkg_rel) };
            let files: Vec<String> = cand.test_files.iter()
                .filter(|f| tracked.contains(*f) && !BUILD_FILES.contains(&super::file_name(f)))
                .filter(|f| package_of(f, root, &tracked).as_deref() == Some(pkg.as_str()))
                .map(|f| label(root, &pkg, Some(f))).collect();
            let found = if files.is_empty() { vec![] } else {
                let (code, log) = self.exec(self.run(commit_tree, cand, true, query_script(unit, &files)))?;
                if code == 0 || code == 3 {
                    log.lines().filter(|l| l.starts_with("//") || l.starts_with("@@//"))
                        .map(str::to_string).collect()
                } else {
                    vec![]
                }
            };
            if found.is_empty() {
                out.insert(unit.clone());
            } else {
                out.extend(found);
            }
        }
        Ok(out.into_iter().collect())
    }

    fn prefetch_label(&self) -> &'static str {
        "bazel fetch (test --nobuild)"
    }

    fn test_label(&self, units: &[String]) -> String {
        format!("bazel test {}", units.join(" "))
    }

    fn prefetch(&self, tree: &Path, cand: &Candidate) -> Result<(bool, String)> {
        super::own_volume(&self.sandbox.image, &self.cache, tree)?;
        let (code, log) = self.exec(self.run(tree, cand, true, prefetch_script(&cand.packages)))?;
        Ok((code == 0, log))
    }

    fn run_tests(&self, tree: &Path, cand: &Candidate) -> Result<(Outcome, String)> {
        let (code, out) = self.exec(self.run(tree, cand, false, test_script(&cand.packages)))?;
        let log = format!("$ {}\nexit={code}\n{out}", self.test_label(&cand.packages));
        Ok((classify(code, &log), log))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bazel_files_classify_build_files_as_tests() {
        assert_eq!(classify_file("pkg/cron/BUILD.bazel"), FileKind::Test);
        assert_eq!(classify_file("pkg/cron/BUILD"), FileKind::Test);
        assert_eq!(classify_file("pkg/cron/cron_test.go"), FileKind::Test);
        assert_eq!(classify_file("tools/x/runner_test.ml"), FileKind::Test);
        assert_eq!(classify_file("tools/x/test_cli.py"), FileKind::Test);
        assert_eq!(classify_file("pkg/cron/testdata/a.json"), FileKind::Test);
        assert_eq!(classify_file("pkg/cron/cron.go"), FileKind::Source);
        assert_eq!(classify_file("tools/x/runner.ml"), FileKind::Source);
        assert_eq!(classify_file("MODULE.bazel"), FileKind::Source);
        assert_eq!(classify_file("bazel/defs.bzl"), FileKind::Source);
        assert_eq!(classify_file("pkg/api.pb.go"), FileKind::Generated);
        assert_eq!(classify_file("pkg/README.md"), FileKind::Other);
        assert!(is_root_marker("candace/MODULE.bazel") && is_root_marker("WORKSPACE"));
        assert!(!is_root_marker("BUILD.bazel"));
    }

    #[test]
    fn labels_and_packages_are_relative_to_the_workspace() {
        let tracked: BTreeSet<String> = ["ws/MODULE.bazel", "ws/pkg/BUILD.bazel", "ws/pkg/sub/a_test.go",
                                         "ws/BUILD.bazel"].map(String::from).into_iter().collect();
        assert_eq!(package_of("ws/pkg/sub/a_test.go", "ws", &tracked).as_deref(), Some("ws/pkg"));
        assert_eq!(label("ws", "ws/pkg", Some("ws/pkg/sub/a_test.go")), "//pkg:sub/a_test.go");
        assert_eq!(label("ws", "ws/pkg", None), "//pkg:all");
        assert_eq!(label("ws", "ws", None), "//:all");
        assert_eq!(label("", "pkg", None), "//pkg:all");
        assert_eq!(package_of("other/x_test.go", "ws", &tracked), None, "outside the workspace");
    }

    #[test]
    fn units_are_test_packages_whose_build_file_names_the_changed_test() {
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        let run = |args: &[&str]| assert!(std::process::Command::new("git").args(args).current_dir(r)
            .env("GIT_AUTHOR_NAME", "t").env("GIT_AUTHOR_EMAIL", "t@example.invalid")
            .env("GIT_COMMITTER_NAME", "t").env("GIT_COMMITTER_EMAIL", "t@example.invalid")
            .status().unwrap().success());
        let write = |p: &str, s: &str| {
            std::fs::create_dir_all(r.join(p).parent().unwrap()).unwrap();
            std::fs::write(r.join(p), s).unwrap();
        };
        run(&["init", "-q"]);
        write("MODULE.bazel", "module(name = \"m\")\n");
        write("pkg/a/BUILD.bazel", "go_test(\n    name = \"a_test\",\n    srcs = [\"a_test.go\"],\n)\n");
        write("pkg/b/BUILD.bazel", "go_library(name = \"b\", srcs = [\"b.go\"])\n");
        write("pkg/a/a.go", "package a\n");
        write("pkg/a/a_test.go", "package a\n");
        write("pkg/b/b_test.go", "package b\n");
        run(&["add", "-A"]);
        run(&["commit", "-qm", "base"]);
        let sha = git(r, &["rev-parse", "HEAD"]).unwrap().trim().to_string();
        let tracked = super::super::tracked_files(r, "HEAD").unwrap().into_iter().collect();
        let ch = Changed { sha, src: vec![(1, "pkg/a/a.go".into())],
                           tests: vec!["pkg/a/a_test.go".into(), "pkg/b/b_test.go".into(), "pkg/a/BUILD.bazel".into()],
                           tracked };
        assert_eq!(BazelMapper.root(&ch).as_deref(), Some(""));
        assert_eq!(BazelMapper.units(&ch, "", r).unwrap(), ["//pkg/a:all"], "pkg/b declares no test rule");
    }

    #[test]
    fn the_scripts_share_one_output_base_and_the_query_drops_manual_targets() {
        let p = prefetch_script(&["//pkg/a:all".into()]);
        assert!(p.starts_with("bazel --output_user_root=/cache/out test --nobuild --keep_going"), "{p}");
        let t = test_script(&["//pkg/a:a_test".into()]);
        assert!(t.contains("--repository_cache=/cache/repos --disk_cache=/cache/disk") && t.ends_with("'//pkg/a:a_test'\n"));
        let q = query_script("//pkg/a:all", &["//pkg/a:a_test.go".into()]);
        assert!(q.contains("tests(rdeps(//pkg/a:all, set(//pkg/a:a_test.go))) except attr(tags, "), "{q}");
    }

    // Regression pins: real Bazel 9 output shapes.

    const EXTERNAL_FETCH: &str = "$ bazel test //pkg/cron:cron_test\nexit=1\n\
        ERROR: /src/pkg/cron/BUILD.bazel:20:8: no such package '@@gazelle++go_deps+com_github_robfig_cron_v3//': \
        java.io.IOException: Error downloading [https://proxy.golang.org/github.com/robfig/cron/v3/@v/v3.0.1.zip] \
        to /cache/out/x/external/gazelle++go_deps+com_github_robfig_cron_v3/temp: Unknown host: proxy.golang.org \
        and referenced by '//pkg/cron:cron_test'\n\
        ERROR: Analysis of target '//pkg/cron:cron_test' failed; build aborted: Analysis failed\n\
        INFO: Build did NOT complete successfully\n";

    const OWN_TARGET_MISSING: &str = "$ bazel test //pkg/warden/election:election_test\nexit=1\n\
        ERROR: /src/pkg/warden/election/BUILD.bazel:30:8: no such package 'pkg/warden/internal/transportidentity': \
        BUILD file not found in any of the following directories. Add a BUILD file to a directory to mark it as a package.\n\
        \x20- /src/pkg/warden/internal/transportidentity and referenced by '//pkg/warden/election:election_test'\n\
        ERROR: Analysis of target '//pkg/warden/election:election_test' failed; build aborted: Analysis failed\n\
        INFO: Build did NOT complete successfully\n";

    const OWN_SOURCE_MISSING: &str = "ERROR: /src/pkg/box/BUILD.bazel:3:11: missing input file '//pkg/box:limit.go'\n\
        ERROR: /src/pkg/box/BUILD.bazel:3:11: 1 input file(s) do not exist\n\
        INFO: Build did NOT complete successfully\n";

    const COMPILE_ERROR: &str = "ERROR: /src/pkg/box/BUILD.bazel:12:8: GoCompilePkg pkg/box/box_test.internal.a \
        failed: (Exit 1): builder failed: error executing GoCompilePkg command (from target //pkg/box:box_test)\n\
        pkg/box/box_test.go:9:5: undefined: NewThing\n\
        INFO: Build did NOT complete successfully\n";

    const TEST_FAILED: &str = "FAIL: //pkg/box:box_test (see /cache/out/x/execroot/_main/bazel-out/k8-fastbuild/testlogs/pkg/box/box_test/test.log)\n\
        //pkg/box:box_test                                                       FAILED in 0.4s\n\n\
        Executed 1 out of 1 test: 1 fails locally.\n";

    #[test]
    fn a_missing_third_party_repository_is_infra() {
        assert_eq!(classify(1, EXTERNAL_FETCH), Outcome::Infra);
        assert_eq!(classify(1, "ERROR: Error computing the main repository mapping: Error accessing registry \
                                https://bcr.bazel.build/: Failed to fetch registry file\n"), Outcome::Infra);
        assert_eq!(classify(1, "ERROR: /src/pkg/x/BUILD.bazel:4:8: no such package '@com_github_new_dep//': \
                                No repository visible as '@com_github_new_dep' from main repository\n"), Outcome::Infra);
        assert_eq!(classify(4, "ERROR: No test targets were found, yet testing was requested\n"), Outcome::Infra);
    }

    #[test]
    fn a_new_package_target_or_file_of_the_repository_missing_in_the_parent_is_a_build_failure() {
        assert_eq!(classify(1, OWN_TARGET_MISSING), Outcome::BuildFail);
        assert_eq!(classify(1, OWN_SOURCE_MISSING), Outcome::BuildFail);
        assert_eq!(classify(1, "ERROR: Skipping '//pkg/new:new_test': no such target '//pkg/new:new_test': \
                                target 'new_test' not declared in package 'pkg/new'\n"), Outcome::BuildFail);
        assert_eq!(classify(1, COMPILE_ERROR), Outcome::BuildFail);
        assert!(blames_own("no such package '@@//pkg/x': BUILD file not found"));
        assert!(!blames_own("no such package '@@rules_go+//go': x"));
        // Own blame never hides an external fetch failure elsewhere.
        assert_eq!(classify(1, &format!("{OWN_TARGET_MISSING}{EXTERNAL_FETCH}")), Outcome::Infra);
    }

    #[test]
    fn test_failures_passes_and_timeouts_classify() {
        assert_eq!(classify(3, TEST_FAILED), Outcome::TestFail);
        assert_eq!(classify(0, "Executed 1 out of 1 test: 1 test passes.\n"), Outcome::Pass);
        assert_eq!(classify(137, ""), Outcome::Timeout);
        assert_eq!(classify(8, "something odd"), Outcome::Infra);
    }
}
