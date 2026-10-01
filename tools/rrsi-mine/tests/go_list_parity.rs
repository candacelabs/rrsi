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

//! Parity: `rrsi-mine list` on a synthetic Go repository prints exactly the
//! bytes it printed before toolchains existed (tests/fixtures/
//! go_list.golden.jsonl was captured from that binary). Every commit's
//! author, committer and date are fixed, so the SHAs are reproducible.

use std::path::Path;
use std::process::Command;

fn git(repo: &Path, date: &str, args: &[&str]) {
    let ok = Command::new("git").args(["-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"])
        .args(args).current_dir(repo)
        .env("GIT_AUTHOR_NAME", "t").env("GIT_AUTHOR_EMAIL", "t@example.invalid")
        .env("GIT_COMMITTER_NAME", "t").env("GIT_COMMITTER_EMAIL", "t@example.invalid")
        .env("GIT_AUTHOR_DATE", date).env("GIT_COMMITTER_DATE", date)
        .status().unwrap().success();
    assert!(ok, "git {args:?}");
}

fn write(repo: &Path, path: &str, text: &str) {
    let p = repo.join(path);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, text).unwrap();
}

/// A repository whose history covers every candidate rule: a plain task, a
/// task in a nested module, a body, test-only and source-only commits, a
/// commit over the churn limit, a commit across two modules, generated code
/// and a non-Go file beside a task.
pub fn synthetic_go_repo(r: &Path) {
    let mut day = 0;
    let mut commit = |msg: &str| {
        day += 1;
        let date = format!("2026-07-{day:02}T12:00:00Z");
        git(r, &date, &["add", "-A"]);
        git(r, &date, &["commit", "-q", "-m", msg]);
    };
    git(r, "2026-07-01T00:00:00Z", &["init", "-q"]);
    write(r, "go.mod", "module example.invalid/root\n\ngo 1.26\n");
    write(r, "pkg/a/a.go", "package a\n");
    write(r, "sub/go.mod", "module example.invalid/sub\n\ngo 1.26\n");
    write(r, "sub/b/b.go", "package b\n");
    commit("base");

    write(r, "pkg/a/a.go", "package a\n\nfunc A() int { return 1 }\n");
    write(r, "pkg/a/a_test.go", "package a\n");
    write(r, "pkg/a/README.md", "a\n");
    commit("feat(a): add A");

    write(r, "sub/b/b.go", "package b\n\nfunc B() int { return 2 }\n");
    write(r, "sub/b/b_test.go", "package b\n");
    commit("feat(b): add B\n\nThe body explains why B exists.\nSecond line.");

    write(r, "pkg/a/more_test.go", "package a\n");
    commit("test: only tests");

    write(r, "pkg/a/a.go", "package a\n\nfunc A() int { return 3 }\n");
    commit("fix: source only");

    let big: String = (0..450).map(|i| format!("var V{i} = {i}\n")).collect();
    write(r, "pkg/a/big.go", &format!("package a\n\n{big}"));
    write(r, "pkg/a/big_test.go", "package a\n");
    commit("feat: too big");

    write(r, "pkg/a/two.go", "package a\n\nvar Two = 2\n");
    write(r, "pkg/a/two_test.go", "package a\n");
    write(r, "sub/b/two.go", "package b\n\nvar Two = 2\n");
    write(r, "sub/b/two_test.go", "package b\n");
    commit("feat: two modules");

    write(r, "pkg/a/gen/x.go", "package gen\n\nvar X = 1\n");
    write(r, "pkg/a/api.pb.go", "package a\n\nvar P = 1\n");
    write(r, "pkg/a/c.go", "package a\n\nfunc C() {}\n");
    write(r, "pkg/a/c_test.go", "package a\n");
    commit("feat(a): C beside generated code");

    write(r, "pkg/c/c.go", "package c\n\nfunc C() {}\n");
    write(r, "pkg/c/c_test.go", "package c\n");
    write(r, "pkg/d/d.go", "package d\n\nfunc D() {}\n");
    write(r, "pkg/d/d_test.go", "package d\n");
    commit("feat: two packages of one module");
}

fn list(repo: &Path, extra: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_rrsi-mine"))
        .args(["list", "--repo"]).arg(repo).args(["--since", "2000-01-01"]).args(extra)
        .output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

const GOLDEN: &str = include_str!("fixtures/go_list.golden.jsonl");

#[test]
fn go_list_output_is_byte_for_byte_unchanged() {
    let d = tempfile::tempdir().unwrap();
    synthetic_go_repo(d.path());
    let got = list(d.path(), &[]);
    if std::env::var_os("RRSI_WRITE_GOLDEN").is_some() {
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/go_list.golden.jsonl"), &got).unwrap();
    }
    assert_eq!(got, GOLDEN);
    assert_eq!(list(d.path(), &["--toolchain", "go"]), GOLDEN);
    assert_eq!(list(d.path(), &["--toolchain", "auto"]), GOLDEN, "auto finds only Go here");
}

/// A Python project beside the Go module: `--toolchain python` lists its
/// commit with the pytest file as the unit, Go's listing is unchanged, and
/// `auto` lists both, each candidate once.
#[test]
fn python_candidates_list_beside_go_ones() {
    let d = tempfile::tempdir().unwrap();
    let r = d.path();
    synthetic_go_repo(r);
    write(r, "py/pyproject.toml", "[project]\nname = \"p\"\n");
    write(r, "py/pkg/__init__.py", "");
    write(r, "py/pkg/core.py", "def one():\n    return 1\n");
    git(r, "2026-08-01T12:00:00Z", &["add", "-A"]);
    git(r, "2026-08-01T12:00:00Z", &["commit", "-q", "-m", "py: base"]);
    write(r, "py/pkg/core.py", "def one():\n    return 1\n\n\ndef two():\n    return 2\n");
    write(r, "py/tests/test_core.py", "from pkg.core import two\n\n\ndef test_two():\n    assert two() == 2\n");
    write(r, "py/tests/helpers.py", "");
    git(r, "2026-08-02T12:00:00Z", &["add", "-A"]);
    git(r, "2026-08-02T12:00:00Z", &["commit", "-q", "-m", "py: add two"]);
    assert_eq!(list(r, &[]), GOLDEN, "Go's listing ignores Python");
    let py: Vec<serde_json::Value> = list(r, &["--toolchain", "python"]).lines()
        .map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(py.len(), 1);
    assert_eq!(py[0]["subject"], "py: add two");
    assert_eq!(py[0]["toolchain"], "python");
    assert_eq!(py[0]["module_root"], "py");
    assert_eq!(py[0]["packages"], serde_json::json!(["./tests/test_core.py"]));
    assert_eq!(py[0]["src_files"], serde_json::json!(["py/pkg/core.py"]));
    assert_eq!(py[0]["test_files"], serde_json::json!(["py/tests/helpers.py", "py/tests/test_core.py"]));
    let auto = list(r, &["--toolchain", "auto"]);
    assert_eq!(auto.lines().count(), GOLDEN.lines().count() + 1);
    assert!(auto.starts_with(GOLDEN), "Go first, unchanged");
}
