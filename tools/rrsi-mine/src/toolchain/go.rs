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

//! Go: `go test` of the packages that change both source and tests, in a
//! network-less `golang` container after `go mod download all`. This is the
//! miner's original behaviour, unchanged: the trait methods delegate to
//! [`crate::candidates`] and [`crate::Docker`].

use super::{FileKind, Toolchain};
use crate::{Candidate, Docker, Outcome, GENERATED};
use anyhow::Result;
use std::path::Path;

/// The Go toolchain's container settings (see [`crate::Docker`]).
pub struct Go {
    pub image: String,
    pub modcache: String,
    pub buildcache: String,
    pub test_timeout: u64,
}

impl Go {
    pub fn docker(&self) -> Docker<'_> {
        Docker { image: &self.image, modcache: &self.modcache, buildcache: &self.buildcache,
                 test_timeout: self.test_timeout }
    }
}

/// The Go miner's file rules: `_test.go` is a test, GENERATED patterns are
/// generated, other `.go` files are source.
pub fn classify_file(path: &str) -> FileKind {
    if !path.ends_with(".go") {
        FileKind::Other
    } else if path.ends_with("_test.go") {
        FileKind::Test
    } else if GENERATED.iter().any(|g| path.contains(g)) {
        FileKind::Generated
    } else {
        FileKind::Source
    }
}

impl Toolchain for Go {
    fn name(&self) -> &'static str {
        "go"
    }

    fn is_root_marker(&self, path: &str) -> bool {
        super::file_name(path) == "go.mod"
    }

    fn classify_file(&self, path: &str) -> FileKind {
        classify_file(path)
    }

    fn candidates(&self, repo: &Path, since: &str) -> Result<Vec<Candidate>> {
        crate::candidates(repo, since)
    }

    fn prefetch_label(&self) -> &'static str {
        "go mod download"
    }

    fn test_label(&self, units: &[String]) -> String {
        format!("go test {}", units.join(" "))
    }

    fn prefetch(&self, tree: &Path, cand: &Candidate) -> Result<(bool, String)> {
        self.docker().download(tree, &cand.module_root)
    }

    fn run_tests(&self, tree: &Path, cand: &Candidate) -> Result<(Outcome, String)> {
        self.docker().go_test(tree, &cand.module_root, &cand.packages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn go() -> Go {
        Go { image: "golang".into(), modcache: "m".into(), buildcache: "b".into(), test_timeout: 1 }
    }

    #[test]
    fn go_files_classify_as_before() {
        let g = go();
        assert_eq!(g.classify_file("pkg/a.go"), FileKind::Source);
        assert_eq!(g.classify_file("pkg/a_test.go"), FileKind::Test);
        assert_eq!(g.classify_file("pkg/api.pb.go"), FileKind::Generated);
        assert_eq!(g.classify_file("pkg/gen/x.go"), FileKind::Generated);
        assert_eq!(g.classify_file("pkg/README.md"), FileKind::Other);
        assert!(g.is_root_marker("sub/go.mod") && !g.is_root_marker("sub/go.sum"));
        assert_eq!(g.test_label(&["./a".into(), "./b".into()]), "go test ./a ./b");
    }

    #[test]
    fn go_roots_are_detected_at_a_revision() {
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        let run = |args: &[&str]| assert!(std::process::Command::new("git").args(args).current_dir(r)
            .env("GIT_AUTHOR_NAME", "t").env("GIT_AUTHOR_EMAIL", "t@example.invalid")
            .env("GIT_COMMITTER_NAME", "t").env("GIT_COMMITTER_EMAIL", "t@example.invalid")
            .status().unwrap().success());
        run(&["init", "-q"]);
        std::fs::create_dir_all(r.join("sub")).unwrap();
        std::fs::write(r.join("go.mod"), "module example.invalid/m\n").unwrap();
        std::fs::write(r.join("sub/go.mod"), "module example.invalid/s\n").unwrap();
        run(&["add", "-A"]);
        run(&["commit", "-qm", "base"]);
        assert_eq!(go().detect(r, "HEAD").unwrap(), ["", "sub"]);
    }
}
