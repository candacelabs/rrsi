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

//! CSF integration: mine any CSF-instrumented Go repository using CSF's own
//! metadata and gates (see tools/rrsi-mine/CSF.md).
//!
//! ```text
//! rrsi-mine csf detect --repo PATH [--rev REV] [--csfc PATH] [--csf-grammar PATH] [--json]
//! rrsi-mine csf guard  --tree DIR  [--csfc PATH] [--csf-grammar PATH]
//! rrsi-mine csf annotate --tasks DIR --repo PATH [--rev REV | --csf-model FILE --csf-root DIR]
//! rrsi-mine mine ... [--csfc PATH] [--csf-grammar PATH] [--csf-model FILE --csf-root DIR]
//! ```
//!
//! The CSF compiler (`csfc`) owns the architecture language: rrsi-mine reads
//! the model only as `csfc emit --format json` prints it and runs csfc's own
//! `check` and `check-generated` gates. Nothing here parses `.csf` files.

pub mod csfc;
pub mod detect;
pub mod generated;
pub mod guard;
pub mod map;
pub mod model;
pub mod source;

use anyhow::{Context, Result};
use model::{Architecture, LocatedModel};
use std::path::{Path, PathBuf};

/// The directory part of a repository-relative path (`""` at the root).
pub fn parent_dir(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

/// `dir` joined with a relative `path`; either may be empty.
pub fn join(dir: &str, path: &str) -> String {
    let path = path.trim_start_matches("./").trim_end_matches('/');
    match (dir.is_empty(), path.is_empty()) {
        (true, _) => path.to_string(),
        (false, true) => dir.to_string(),
        (false, false) => format!("{dir}/{path}"),
    }
}

/// Whether `path` is `prefix` itself or lies below it.
pub fn under(path: &str, prefix: &str) -> bool {
    prefix.is_empty() || path == prefix
        || (path.len() > prefix.len() && path.starts_with(prefix) && path.as_bytes()[prefix.len()] == b'/')
}

/// How `mine` and `csf annotate` use CSF for one repository: the model the
/// tasks are mapped onto, its generated roots, and how to run the guards.
#[derive(Default)]
pub struct MineCsf {
    /// A CSF signal fired (see [`detect`]); without one nothing below is used.
    pub instrumented: bool,
    pub models: Vec<LocatedModel>,
    /// Where the models came from: a revision, or `file:<path>`.
    pub model_rev: Option<String>,
    pub generated_paths: Vec<String>,
    /// Guard configuration; guards run on each commit tree when instrumented.
    pub csfc: Option<PathBuf>,
    pub grammar: Option<PathBuf>,
    pub sources: Vec<String>,
    /// Why no model is available, for the log.
    pub note: String,
}

/// A clean checkout of `rev`, for csfc (which reads files, not git).
pub fn materialize(repo: &Path, rev: &str) -> Result<tempfile::TempDir> {
    let dir = tempfile::Builder::new().prefix("rrsi-csf-").tempdir()?;
    crate::export_tree(repo, rev, dir.path()).with_context(|| format!("exporting {rev}"))?;
    Ok(dir)
}

impl MineCsf {
    /// Detect CSF in `repo` at `rev`, then take the model from `model_file`
    /// (a `csfc emit --format json` document whose paths are relative to
    /// `model_root`) or, failing that, by running csfc on a checkout of `rev`.
    pub fn resolve(repo: &Path, rev: &str, csfc_flag: Option<&Path>, grammar: Option<&Path>,
                   sources: &[String], model_file: Option<(&Path, &str)>) -> Result<Self> {
        let src = source::Source::git(repo, rev)?;
        let (mut d, files) = detect::detect(&src, sources)?;
        let mut out = MineCsf { instrumented: d.instrumented, csfc: csfc_flag.map(Path::to_path_buf),
                                grammar: grammar.map(Path::to_path_buf), sources: sources.to_vec(),
                                ..Default::default() };
        if let Some((file, root)) = model_file {
            out.models = vec![LocatedModel { path: None, root: root.to_string(),
                                             model: Architecture::from_file(file)? }];
            out.model_rev = Some(format!("file:{}", file.display()));
            out.instrumented = true;
        } else if d.instrumented && d.csf_files.iter().any(|f| f.status != "other") {
            let info = csfc::locate(csfc_flag);
            if info.available {
                let tree = materialize(repo, rev)?;
                detect::with_csfc(&mut d, &files, tree.path(), info, grammar)?;
                out.models = d.models.clone();
                out.model_rev = Some(src.rev_id().unwrap_or_else(|| rev.to_string()));
                out.note = d.csf_files.iter().filter(|f| f.status != "architecture_model" && f.status != "other")
                    .map(|f| format!("{}: {} {}", f.path, f.status, f.detail)).collect::<Vec<_>>().join("; ");
            } else {
                out.note = info.detail;
            }
        } else if d.instrumented {
            out.note = format!("no architecture source (<root>/{}) at {rev}", csfc::MODEL_SOURCE);
        }
        out.generated_paths = out.models.iter().flat_map(LocatedModel::generated_paths).collect();
        out.generated_paths.sort();
        out.generated_paths.dedup();
        Ok(out)
    }

    /// The task's `csf` field: `None` for a repository with no CSF signal.
    pub fn task(&self, src_files: &[String]) -> Option<map::TaskCsf> {
        self.instrumented.then(|| map::map_files(&self.models, src_files, self.model_rev.as_deref()))
    }

    /// Run the guards on a commit tree, marking which bind the agent.
    pub fn guards(&self, tree: &Path) -> Result<Option<Vec<guard::GuardVerdict>>> {
        if !self.instrumented {
            return Ok(None);
        }
        let mut v = guard::guard(tree, self.csfc.as_deref(), self.grammar.as_deref(), &self.sources)?;
        guard::mark_required(&mut v);
        Ok(Some(v))
    }

    /// One line for the miner's log.
    pub fn describe(&self) -> String {
        if !self.instrumented {
            return "CSF: not instrumented".into();
        }
        let names: Vec<String> = self.models.iter()
            .map(|m| format!("{} ({} components, root {:?})", m.model.architecture.name, m.model.components.len(), m.root))
            .collect();
        match names.is_empty() {
            true => format!("CSF: instrumented, no model ({}); tasks get csf with no components", self.note),
            false => format!("CSF: model {} at {}; {} generated roots excluded from source",
                             names.join(", "), self.model_rev.as_deref().unwrap_or("?"), self.generated_paths.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_helpers() {
        assert_eq!(parent_dir("a/b/c.go"), "a/b");
        assert_eq!(parent_dir("c.go"), "");
        assert_eq!(join("", "x/y"), "x/y");
        assert_eq!(join("root", "./x/y/"), "root/x/y");
        assert_eq!(join("root", ""), "root");
        assert!(under("a/b/c.go", "a/b"));
        assert!(under("a/b", "a/b"));
        assert!(!under("a/bc/d.go", "a/b"), "a prefix must end at a path separator");
        assert!(under("anything", ""));
    }

    #[test]
    fn a_repository_without_csf_gets_no_csf_field() {
        let (d, _) = source::tests::repo_with(&[&[("go.mod", "module example.invalid/plain\n"), ("a.go", "package a\n")]]);
        let m = MineCsf::resolve(d.path(), "HEAD", None, None, &[], None).unwrap();
        assert!(!m.instrumented);
        assert_eq!(m.task(&["a.go".into()]), None);
        assert_eq!(m.guards(d.path()).unwrap(), None);
        assert_eq!(m.describe(), "CSF: not instrumented");
    }

    #[test]
    fn a_model_file_maps_tasks_and_declares_generated_roots() {
        let (d, _) = source::tests::repo_with(&[&[("svc/go.mod", "module example.invalid/svc\n\n\
                                                    require github.com/candacelabs/csf v0.1.0\n")]]);
        let json = d.path().join("shop.json");
        std::fs::write(&json, model::tests::SHOP).unwrap();
        let m = MineCsf::resolve(d.path(), "HEAD", None, None, &[], Some((&json, "svc"))).unwrap();
        assert!(m.instrumented);
        assert_eq!(m.generated_paths, ["svc/csf/architecture/generated", "svc/services/orders/api_cgen.go"]);
        let t = m.task(&["svc/pkg/money/a.go".into()]).unwrap();
        assert_eq!(t.components, ["money"]);
        assert!(t.model_rev.unwrap().starts_with("file:"));
        assert!(m.describe().contains("shop (6 components"));
    }

    #[test]
    fn an_instrumented_repository_without_csfc_still_gets_a_csf_field() {
        let (d, _) = source::tests::repo_with(&[&[("go.mod", "module github.com/candacelabs/csf\n"),
                                                  ("csf/architecture/architecture.csf", "x")]]);
        let m = MineCsf::resolve(d.path(), "HEAD", Some(&d.path().join("no-csfc")), None, &[], None).unwrap();
        assert!(m.instrumented && m.models.is_empty());
        assert!(m.note.starts_with("cannot run"), "{}", m.note);
        let t = m.task(&["a.go".into()]).unwrap();
        assert!(t.components.is_empty() && t.models.is_empty());
    }
}
