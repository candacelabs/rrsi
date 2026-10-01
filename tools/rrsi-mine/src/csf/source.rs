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

//! Files of one source tree, read without a checkout: a git revision through
//! gitoxide (`gix`, in-process, no `git` subprocess), or a directory through
//! ripgrep's `ignore` walker (honours .gitignore, never follows symlinks).

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub enum Source {
    Git { repo: Box<gix::Repository>, commit: gix::ObjectId, tree: gix::ObjectId },
    Dir(PathBuf),
}

impl Source {
    /// The tree of `rev` (any revision git understands) in the repository
    /// containing `repo`.
    pub fn git(repo: &Path, rev: &str) -> Result<Self> {
        let r = gix::discover(repo).with_context(|| format!("opening git repository {}", repo.display()))?;
        Self::git_in(&r, rev)
    }

    /// Like [`Source::git`] for an already opened repository (cheap to
    /// repeat per commit).
    pub fn git_in(repo: &gix::Repository, rev: &str) -> Result<Self> {
        let commit = repo.rev_parse_single(rev).with_context(|| format!("resolving {rev}"))?
            .object()?.peel_to_commit().with_context(|| format!("{rev} is not a commit"))?;
        let tree = commit.tree_id().map_err(|e| anyhow::anyhow!("{rev} has no tree: {e}"))?.detach();
        Ok(Source::Git { repo: Box::new(repo.clone()), commit: commit.id, tree })
    }

    /// The commit id of a git source.
    pub fn rev_id(&self) -> Option<String> {
        match self {
            Source::Git { commit, .. } => Some(commit.to_string()),
            Source::Dir(_) => None,
        }
    }

    pub fn dir(root: &Path) -> Self {
        Source::Dir(root.to_path_buf())
    }

    /// Every regular file of a directory tree, ignoring `.gitignore` and
    /// similar rules (only `.git` is skipped). For exported or patched
    /// checkouts, where a tracked file matching an ignore rule (e.g. a
    /// force-added `architecture.csf` under a `*.csf` rule) must still count.
    pub fn all_files(&self) -> Result<Vec<String>> {
        let Source::Dir(root) = self else { return self.files() };
        let mut v = Vec::new();
        let walk = ignore::WalkBuilder::new(root).standard_filters(false)
            .filter_entry(|e| e.file_name() != ".git").build();
        for entry in walk {
            let entry = entry.context("walking the directory")?;
            if entry.file_type().is_some_and(|t| t.is_file()) {
                let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
                v.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
        v.sort();
        Ok(v)
    }

    /// Every regular file, repository-relative with `/` separators, sorted.
    pub fn files(&self) -> Result<Vec<String>> {
        let mut out = match self {
            Source::Git { repo, tree, .. } => {
                let mut rec = gix::traverse::tree::Recorder::default();
                repo.find_tree(*tree)?.traverse().breadthfirst(&mut rec).context("walking the git tree")?;
                rec.records.into_iter().filter(|e| e.mode.is_blob())
                    .map(|e| e.filepath.to_string()).collect()
            }
            Source::Dir(root) => {
                let mut v = Vec::new();
                let walk = ignore::WalkBuilder::new(root).hidden(false).require_git(false)
                    .filter_entry(|e| e.file_name() != ".git").build();
                for entry in walk {
                    let entry = entry.context("walking the directory")?;
                    if entry.file_type().is_some_and(|t| t.is_file()) {
                        let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
                        v.push(rel.to_string_lossy().replace('\\', "/"));
                    }
                }
                v
            }
        };
        out.sort();
        Ok(out)
    }

    /// The bytes of one repository-relative file, `None` when absent.
    pub fn read(&self, path: &str) -> Result<Option<Vec<u8>>> {
        match self {
            Source::Git { repo, tree, .. } => {
                let tree = repo.find_tree(*tree)?;
                match tree.lookup_entry_by_path(path)? {
                    Some(entry) if entry.mode().is_blob() => Ok(Some(entry.object()?.detach().data)),
                    _ => Ok(None),
                }
            }
            Source::Dir(root) => match std::fs::read(root.join(path)) {
                Ok(b) => Ok(Some(b)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(e).with_context(|| format!("reading {path}")),
            },
        }
    }

    pub fn read_string(&self, path: &str) -> Result<Option<String>> {
        Ok(self.read(path)?.map(|b| String::from_utf8_lossy(&b).into_owned()))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #[test]
    fn all_files_sees_tracked_files_that_match_ignore_rules() {
        // Review (P2): the ignore-aware walk hid a force-added
        // architecture.csf matching a `*.csf` rule, so guards were skipped.
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(".gitignore"), "*.csf\n").unwrap();
        std::fs::create_dir_all(d.path().join("csf/architecture")).unwrap();
        std::fs::write(d.path().join(crate::csf::csfc::MODEL_SOURCE), "m").unwrap();
        std::fs::create_dir(d.path().join(".git")).unwrap();
        std::fs::write(d.path().join(".git/HEAD"), "x").unwrap();
        let src = Source::dir(d.path());
        assert!(src.all_files().unwrap().contains(&crate::csf::csfc::MODEL_SOURCE.to_string()));
        assert!(!src.files().unwrap().contains(&crate::csf::csfc::MODEL_SOURCE.to_string()),
                "the ignore-aware listing hides it, which is why guards must not use it");
        assert!(!src.all_files().unwrap().iter().any(|f| f.starts_with(".git/")));
    }

    #[test]
    fn guard_finds_an_ignored_architecture_source() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(".gitignore"), "*.csf\n").unwrap();
        std::fs::create_dir_all(d.path().join("csf/architecture")).unwrap();
        std::fs::write(d.path().join(crate::csf::csfc::MODEL_SOURCE), "m").unwrap();
        let missing = d.path().join("no-such-csfc");
        let v = crate::csf::guard::guard(d.path(), Some(&missing), None, &[]).unwrap();
        assert!(!v.is_empty());
        assert!(v.iter().all(|g| !g.reason.contains("no architecture source")),
                "the model must be found even though .gitignore matches it: {:?}",
                v.iter().map(|g| &g.reason).collect::<Vec<_>>());
    }

    use super::*;
    use std::process::Command;

    /// A git repository in a temp dir with one commit per `commits` entry.
    /// Returns the directory and each commit's sha.
    pub fn repo_with(commits: &[&[(&str, &str)]]) -> (tempfile::TempDir, Vec<String>) {
        let d = tempfile::tempdir().unwrap();
        let git = |args: &[&str]| {
            let o = Command::new("git").args(args).current_dir(d.path())
                .env("GIT_AUTHOR_NAME", "t").env("GIT_AUTHOR_EMAIL", "t@example.invalid")
                .env("GIT_COMMITTER_NAME", "t").env("GIT_COMMITTER_EMAIL", "t@example.invalid")
                .output().unwrap();
            assert!(o.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&o.stderr));
            String::from_utf8_lossy(&o.stdout).trim().to_string()
        };
        git(&["init", "-q"]);
        let mut shas = Vec::new();
        for (i, files) in commits.iter().enumerate() {
            for (p, text) in files.iter() {
                let path = d.path().join(p);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, text).unwrap();
            }
            git(&["add", "-A"]);
            git(&["commit", "-qm", &format!("c{i}")]);
            shas.push(git(&["rev-parse", "HEAD"]));
        }
        (d, shas)
    }

    #[test]
    fn a_git_revision_is_read_without_a_checkout() {
        let (d, shas) = repo_with(&[&[("a/x.go", "one")], &[("a/x.go", "two"), ("b/y.go", "y")]]);
        let old = Source::git(d.path(), &shas[0]).unwrap();
        assert_eq!(old.files().unwrap(), ["a/x.go"]);
        assert_eq!(old.read_string("a/x.go").unwrap().as_deref(), Some("one"));
        assert_eq!(old.read("b/y.go").unwrap(), None);
        assert_eq!(old.read("a").unwrap(), None, "a directory is not a file");
        let head = Source::git(d.path(), "HEAD").unwrap();
        assert_eq!(head.rev_id().as_deref(), Some(shas[1].as_str()));
        assert_eq!(head.files().unwrap(), ["a/x.go", "b/y.go"]);
        assert_eq!(head.read_string("a/x.go").unwrap().as_deref(), Some("two"));
        assert!(Source::git(d.path(), "no-such-rev").is_err());
    }

    #[test]
    fn a_directory_walk_skips_git_ignored_files_and_symlinks() {
        let d = tempfile::tempdir().unwrap();
        for (f, text) in [("a/x.go", "x"), (".git/HEAD", "h"), (".gitignore", "out/\n"), ("out/o.go", "o"),
                          (".github/w.yml", "w")] {
            let p = d.path().join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, text).unwrap();
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(d.path().join("a"), d.path().join("bazel-bin")).unwrap();
        let s = Source::dir(d.path());
        assert_eq!(s.files().unwrap(), [".github/w.yml", ".gitignore", "a/x.go"]);
        assert_eq!(s.read_string("a/x.go").unwrap().as_deref(), Some("x"));
        assert_eq!(s.read("missing").unwrap(), None);
    }
}
