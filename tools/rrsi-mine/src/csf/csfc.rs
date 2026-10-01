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

//! Running CSF's compiler, `csfc`, and reading what it prints.
//!
//! csfc is built from CSF's own sources (`csf/compiler/architecture`, Bazel
//! target `//csf/compiler/architecture:csfc`). It is found by `--csfc PATH`,
//! then `RRSI_CSFC`, then a `csfc` on PATH. At run time it also reads CSF's
//! grammar (`language.ebnf`): `--csf-grammar PATH`, then `RRSI_CSF_GRAMMAR`,
//! then `<model root>/csf/compiler/architecture/language.ebnf` when the tree
//! is a CSF checkout itself.

use super::model::{Architecture, LocatedModel};
use super::{join, parent_dir};
use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;
use wait_timeout::ChildExt;

/// Longest one csfc run may take.
pub const CSFC_TIMEOUT: Duration = Duration::from_secs(300);
/// Where a CSF checkout keeps the grammar csfc reads, relative to its root.
pub const GRAMMAR_IN_CSF: &str = "csf/compiler/architecture/language.ebnf";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsfcInfo {
    pub path: Option<String>,
    /// `flag`, `env` (RRSI_CSFC), `path` (a `csfc` on PATH) or `none`.
    pub source: String,
    /// The binary ran (`csfc --help=plain` exited 0).
    pub available: bool,
    pub detail: String,
}

/// One csfc diagnostic, `file:line:col: CODE: message` on its stderr.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub file: String,
    pub line: u64,
    pub col: u64,
    pub code: String,
    pub message: String,
}

fn diagnostic_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+): (?P<code>[A-Za-z][A-Za-z0-9_]*): (?P<message>.*)$")
        .expect("static regex"))
}

/// csfc's stderr as diagnostics. A line that is not in csfc's format is kept
/// whole with code `UNPARSED`, so nothing it said is dropped.
pub fn parse_diagnostics(stderr: &str) -> Vec<Diagnostic> {
    stderr.lines().filter(|l| !l.trim().is_empty()).map(|l| match diagnostic_re().captures(l) {
        Some(c) => Diagnostic {
            file: c["file"].to_string(), line: c["line"].parse().unwrap_or(0), col: c["col"].parse().unwrap_or(0),
            code: c["code"].to_string(), message: c["message"].to_string(),
        },
        None => Diagnostic { file: String::new(), line: 0, col: 0, code: "UNPARSED".into(), message: l.to_string() },
    }).collect()
}

/// `path` made absolute against `cwd` when it names a location relative to
/// it. csfc runs with its working directory set to a temporary checkout, so a
/// relative path the user typed must be resolved before that. A bare program
/// name (no separator) is left alone so it is still looked up on PATH.
pub fn absolutize(path: &Path, cwd: &Path) -> PathBuf {
    if path.is_absolute() || path.components().count() < 2 {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn user_path(path: PathBuf) -> PathBuf {
    match std::env::current_dir() {
        Ok(cwd) => absolutize(&path, &cwd),
        Err(_) => path,
    }
}

/// Find csfc: `flag`, else RRSI_CSFC, else `csfc` on PATH.
pub fn locate(flag: Option<&Path>) -> CsfcInfo {
    let (path, source): (Option<PathBuf>, &str) = match flag {
        Some(p) => (Some(user_path(p.to_path_buf())), "flag"),
        None => match std::env::var_os("RRSI_CSFC").filter(|v| !v.is_empty()) {
            Some(v) => (Some(user_path(PathBuf::from(v))), "env"),
            None => match std::env::var_os("PATH").and_then(|p| {
                std::env::split_paths(&p).map(|d| d.join("csfc")).find(|c| c.is_file())
            }) {
                Some(p) => (Some(p), "path"),
                None => (None, "none"),
            },
        },
    };
    let Some(p) = path else {
        return CsfcInfo { path: None, source: source.into(), available: false,
                          detail: "no csfc: pass --csfc PATH or set RRSI_CSFC".into() };
    };
    let (available, detail) = match Command::new(&p).arg("--help=plain").stdout(Stdio::null())
        .stderr(Stdio::null()).status() {
        Ok(s) if s.success() => (true, "ran --help=plain".to_string()),
        Ok(s) => (false, format!("--help=plain exited {}", s.code().unwrap_or(-1))),
        Err(e) => (false, format!("cannot run: {e}")),
    };
    CsfcInfo { path: Some(p.display().to_string()), source: source.into(), available, detail }
}

impl CsfcInfo {
    pub fn binary(&self) -> Option<PathBuf> {
        self.available.then(|| self.path.as_ref().map(PathBuf::from)).flatten()
    }
}

/// The grammar csfc needs for a model rooted at `tree/root`.
pub fn grammar(flag: Option<&Path>, tree: &Path, root: &str) -> Option<PathBuf> {
    if let Some(p) = flag {
        return Some(user_path(p.to_path_buf()));
    }
    if let Some(v) = std::env::var_os("RRSI_CSF_GRAMMAR").filter(|v| !v.is_empty()) {
        return Some(user_path(PathBuf::from(v)));
    }
    let own = tree.join(root).join(GRAMMAR_IN_CSF);
    own.is_file().then_some(own)
}

pub struct Run {
    /// `None` when csfc was killed by the timeout.
    pub exit: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// Run csfc with `args` in `cwd`, bounded by [`CSFC_TIMEOUT`]. Output goes
/// through temp files so a large diagnostic list cannot fill a pipe.
pub fn run(csfc: &Path, cwd: &Path, args: &[&str]) -> Result<Run> {
    run_with_timeout(csfc, cwd, args, CSFC_TIMEOUT)
}

pub fn run_with_timeout(csfc: &Path, cwd: &Path, args: &[&str], limit: Duration) -> Result<Run> {
    let out = tempfile::tempfile()?;
    let err = tempfile::tempfile()?;
    let mut child = Command::new(csfc).args(args).current_dir(cwd)
        .stdin(Stdio::null()).stdout(out.try_clone()?).stderr(err.try_clone()?)
        .spawn().with_context(|| format!("running {}", csfc.display()))?;
    let exit = match child.wait_timeout(limit)? {
        Some(status) => Some(status.code().unwrap_or(-1)),
        None => {
            child.kill().ok();
            child.wait().ok();
            None
        }
    };
    let read = |mut f: std::fs::File| -> Result<String> {
        use std::io::{Read, Seek};
        f.rewind()?;
        let mut b = Vec::new();
        f.read_to_end(&mut b)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    };
    Ok(Run { exit, stdout: read(out)?, stderr: read(err)? })
}

/// What csfc said about one `.csf` file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsfFile {
    /// Repository-relative path.
    pub path: String,
    /// csfc's `--root`, repository-relative.
    pub root: String,
    /// `architecture_model` (csfc checked it and printed it), `rejected`
    /// (csfc refused it: a documentation vocabulary, a syntax or source-check
    /// error) or `unchecked` (no csfc or no grammar; see `detail`).
    pub status: String,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<Diagnostic>,
}

/// Where csfc looks for an architecture by default, relative to its root.
pub const MODEL_SOURCE: &str = "csf/architecture/architecture.csf";

/// Whether `path` is an architecture source by CSF's layout convention
/// (`<root>/csf/architecture/architecture.csf`, csfc's default `--source`).
/// Other `.csf` files, such as CSF's documentation vocabulary, are not
/// checked unless named with `--csf-source`.
pub fn is_model_source(path: &str) -> bool {
    path == MODEL_SOURCE || path.ends_with(&format!("/{MODEL_SOURCE}"))
}

/// csfc's `--root` for a model at `model_path`: the directory holding
/// `csf/architecture/architecture.csf` by convention; for any other path the
/// nearest ancestor holding a MODULE.bazel, else a go.mod, else the
/// repository root. `files` must be sorted.
pub fn model_root(model_path: &str, files: &[String]) -> String {
    if is_model_source(model_path) {
        return model_path[..model_path.len() - MODEL_SOURCE.len()].trim_end_matches('/').to_string();
    }
    let has = |dir: &str, name: &str| files.binary_search(&join(dir, name)).is_ok();
    let mut ancestors = vec![parent_dir(model_path)];
    while let Some(d) = ancestors.last().filter(|d| !d.is_empty()) {
        ancestors.push(parent_dir(d));
    }
    ancestors.iter().find(|d| has(d, "MODULE.bazel"))
        .or_else(|| ancestors.iter().find(|d| has(d, "go.mod")))
        .map(|d| d.to_string()).unwrap_or_default()
}

/// Ask csfc for the model of `csf_path` in the checkout `tree`
/// (`csfc emit --format json`, which checks first and writes nothing).
pub fn emit_json(csfc: &Path, tree: &Path, csf_path: &str, root: &str, grammar: &Path)
    -> Result<std::result::Result<LocatedModel, (String, Vec<Diagnostic>)>> {
    let rel = csf_path.strip_prefix(root).map(|s| s.trim_start_matches('/')).unwrap_or(csf_path);
    let g = grammar.to_string_lossy();
    let r = run(csfc, &tree.join(root), &["emit", "--format", "json", "--source", rel, "--root", ".", "--grammar", &g])?;
    Ok(match r.exit {
        Some(0) => match Architecture::from_json(&r.stdout) {
            Ok(model) => Ok(LocatedModel { path: Some(csf_path.to_string()), root: root.to_string(), model }),
            Err(e) => Err((format!("csfc printed no csf-architecture JSON: {e:#}"), vec![])),
        },
        Some(1) => Err(("csfc rejected it".into(), parse_diagnostics(&r.stderr))),
        Some(124) if r.stderr.contains("unknown option '--format'") =>
            Err(("this csfc predates `emit --format json`; build CSF's current compiler".into(), vec![])),
        Some(code) => Err((format!("csfc exited {code}"), parse_diagnostics(&r.stderr))),
        None => Err((format!("csfc timed out after {}s", CSFC_TIMEOUT.as_secs()), vec![])),
    })
}

/// The architecture models among `csf_files` of the checkout `tree`, via
/// csfc. Every file gets a status; accepted ones also yield a model.
pub fn models(csfc: Option<&Path>, grammar_flag: Option<&Path>, tree: &Path, csf_files: &[String],
              all_files: &[String]) -> Result<(Vec<CsfFile>, Vec<LocatedModel>)> {
    let mut statuses = Vec::new();
    let mut models = Vec::new();
    for f in csf_files {
        let root = model_root(f, all_files);
        let mut status = CsfFile { path: f.clone(), root: root.clone(), status: "unchecked".into(),
                                   detail: String::new(), diagnostics: vec![] };
        match (csfc, grammar(grammar_flag, tree, &root)) {
            (None, _) => status.detail = "no csfc: pass --csfc PATH or set RRSI_CSFC".into(),
            (Some(_), None) => status.detail = format!(
                "no CSF grammar: pass --csf-grammar PATH or set RRSI_CSF_GRAMMAR ({GRAMMAR_IN_CSF} of a CSF checkout)"),
            (Some(c), Some(g)) => match emit_json(c, tree, f, &root, &g)? {
                Ok(m) => {
                    status.status = "architecture_model".into();
                    status.detail = format!("architecture {} version {}: {} components",
                        m.model.architecture.name, m.model.architecture.version, m.model.components.len());
                    models.push(m);
                }
                Err((detail, diagnostics)) => {
                    status.status = "rejected".into();
                    status.detail = detail;
                    status.diagnostics = diagnostics;
                }
            },
        }
        statuses.push(status);
    }
    Ok((statuses, models))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An executable shell script standing in for csfc. A child `sh` writes
    /// it, so this multi-threaded test process never holds a write handle
    /// that a concurrently forked test could inherit (which makes exec fail
    /// with ETXTBSY, "text file busy").
    #[cfg(unix)]
    pub fn fake_csfc(dir: &Path, body: &str) -> PathBuf {
        use std::io::Write;
        let p = dir.join(format!("csfc-{}", COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
        let mut child = Command::new("sh").args(["-c", "cat > \"$1\" && chmod 755 \"$1\"", "sh"]).arg(&p)
            .stdin(Stdio::piped()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(format!("#!/bin/sh\n{body}\n").as_bytes()).unwrap();
        assert!(child.wait().unwrap().success());
        p
    }

    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    #[test]
    fn diagnostics_are_parsed_and_nothing_is_dropped() {
        let d = parse_diagnostics("csf/architecture/architecture.csf:38:3: CSF_PATH: source or module symlinks are not supported\n\
                                   a:b.csf:2:1: CSF_MODEL: x: y\n\nUsage: csfc [--help]\n");
        assert_eq!(d.len(), 3);
        assert_eq!(d[0], Diagnostic { file: "csf/architecture/architecture.csf".into(), line: 38, col: 3,
                                      code: "CSF_PATH".into(), message: "source or module symlinks are not supported".into() });
        assert_eq!((d[1].file.as_str(), d[1].message.as_str()), ("a:b.csf", "x: y"));
        assert_eq!((d[2].code.as_str(), d[2].message.as_str()), ("UNPARSED", "Usage: csfc [--help]"));
    }

    #[test]
    fn model_root_prefers_bazel_module_then_go_module_then_repo_root() {
        let files = |v: &[&str]| { let mut v: Vec<String> = v.iter().map(|s| s.to_string()).collect(); v.sort(); v };
        assert_eq!(model_root("a/b/m.csf", &files(&["a/MODULE.bazel", "a/b/go.mod"])), "a");
        assert_eq!(model_root("a/b/m.csf", &files(&["a/b/go.mod"])), "a/b");
        assert_eq!(model_root("a/b/m.csf", &files(&[])), "");
        assert_eq!(model_root("m.csf", &files(&["MODULE.bazel"])), "");
        assert_eq!(model_root("x/csf/architecture/architecture.csf", &files(&["MODULE.bazel"])), "x");
        assert_eq!(model_root("csf/architecture/architecture.csf", &files(&[])), "");
        assert!(is_model_source("a/csf/architecture/architecture.csf"));
        assert!(!is_model_source("a/csf/compiler/language/architecture.csf"));
        assert!(!is_model_source("a/xcsf/architecture/architecture.csf"));
    }

    #[cfg(unix)]
    #[test]
    fn csfc_is_located_and_probed() {
        let d = tempfile::tempdir().unwrap();
        let c = locate(Some(&d.path().join("nope")));
        assert_eq!((c.source.as_str(), c.available), ("flag", false));
        assert!(c.detail.starts_with("cannot run"), "{}", c.detail);
        let ok = fake_csfc(d.path(), "exit 0");
        assert!(locate(Some(&ok)).available);
        assert_eq!(locate(Some(&ok)).binary(), Some(ok.clone()));
        let bad = fake_csfc(d.path(), "exit 3");
        let c = locate(Some(&bad));
        assert_eq!((c.available, c.detail.as_str()), (false, "--help=plain exited 3"));
        assert_eq!(c.binary(), None);
    }

    #[cfg(unix)]
    #[test]
    fn a_hung_csfc_is_killed() {
        let d = tempfile::tempdir().unwrap();
        let c = fake_csfc(d.path(), "sleep 30");
        let r = run_with_timeout(&c, d.path(), &[], Duration::from_millis(200)).unwrap();
        assert_eq!(r.exit, None);
    }

    #[cfg(unix)]
    #[test]
    fn models_come_from_csfc_json_and_every_file_gets_a_status() {
        let d = tempfile::tempdir().unwrap();
        let tree = d.path().join("tree");
        std::fs::create_dir_all(tree.join("svc")).unwrap();
        std::fs::write(d.path().join("shop.json"), super::super::model::tests::SHOP).unwrap();
        let grammar = d.path().join("language.ebnf");
        std::fs::write(&grammar, "").unwrap();
        // Prints the fixture for svc/arch.csf, rejects anything else.
        let csfc = fake_csfc(d.path(), &format!(
            "case \"$*\" in *'--source arch.csf'*) cat {};; *) echo 'doc.csf:1:1: CSF_SYNTAX: expected architecture' >&2; exit 1;; esac",
            d.path().join("shop.json").display()));
        let files: Vec<String> = ["doc.csf", "svc/arch.csf", "svc/go.mod"].iter().map(|s| s.to_string()).collect();
        let csf: Vec<String> = files[..2].to_vec();
        let (statuses, models) = models(Some(&csfc), Some(&grammar), &tree, &csf, &files).unwrap();
        assert_eq!(statuses[0].status, "rejected");
        assert_eq!(statuses[0].diagnostics[0].code, "CSF_SYNTAX");
        assert_eq!((statuses[1].status.as_str(), statuses[1].root.as_str()), ("architecture_model", "svc"));
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].path.as_deref(), Some("svc/arch.csf"));

        let (statuses, models) = models_without_csfc(&tree, &csf, &files);
        assert!(models.is_empty() && statuses.iter().all(|s| s.status == "unchecked"));

        let old = fake_csfc(d.path(), "echo \"csfc: unknown option '--format'\" >&2; exit 124");
        let (statuses, _) = super::models(Some(&old), Some(&grammar), &tree, &csf[1..], &files).unwrap();
        assert!(statuses[0].detail.contains("predates"), "{}", statuses[0].detail);
    }

    fn models_without_csfc(tree: &Path, csf: &[String], files: &[String]) -> (Vec<CsfFile>, Vec<LocatedModel>) {
        models(None, None, tree, csf, files).unwrap()
    }

    #[test]
    fn user_supplied_relative_paths_are_resolved_before_csfc_changes_directory() {
        // Review (P1): a relative --csfc/--csf-grammar was probed from the
        // caller's directory but run from inside the temporary checkout.
        let cwd = Path::new("/work/repo");
        assert_eq!(absolutize(Path::new("bazel-bin/csf/csfc.exe"), cwd),
                   PathBuf::from("/work/repo/bazel-bin/csf/csfc.exe"));
        assert_eq!(absolutize(Path::new("./csfc"), cwd), PathBuf::from("/work/repo/./csfc"));
        assert_eq!(absolutize(Path::new("/opt/csfc"), cwd), PathBuf::from("/opt/csfc"));
        // A bare name is still looked up on PATH.
        assert_eq!(absolutize(Path::new("csfc"), cwd), PathBuf::from("csfc"));
    }

    #[test]
    fn grammar_comes_from_the_flag_or_the_csf_checkout() {
        let d = tempfile::tempdir().unwrap();
        assert_eq!(grammar(Some(Path::new("/g.ebnf")), d.path(), ""), Some(PathBuf::from("/g.ebnf")));
        let rel = grammar(Some(Path::new("csf/grammar.ebnf")), d.path(), "").unwrap();
        assert!(rel.is_absolute(), "relative grammar must be resolved: {rel:?}");
        if std::env::var_os("RRSI_CSF_GRAMMAR").is_none() {
            assert_eq!(grammar(None, d.path(), "root"), None);
            let own = d.path().join("root").join(GRAMMAR_IN_CSF);
            std::fs::create_dir_all(own.parent().unwrap()).unwrap();
            std::fs::write(&own, "").unwrap();
            assert_eq!(grammar(None, d.path(), "root"), Some(own));
        }
    }
}
