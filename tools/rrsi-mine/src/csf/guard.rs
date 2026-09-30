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

//! CSF's own gates, run on a checkout as non-compensatory guards: a patch
//! must pass every required guard whatever its test score.
//!
//! For each architecture source in the tree (see [`super::csfc::is_model_source`]
//! and `--csf-source`) two gates run, read-only:
//!
//! | gate | command (in the model's root) | skipped when |
//! |---|---|---|
//! | `csfc check` | `csfc check --source S --root . --grammar G` | no csfc, or no grammar |
//! | `csfc check-generated` | `csfc check-generated --source S --root . --grammar G --output <dir of S>/generated` | as above, or that projection directory does not exist |
//!
//! Status is `pass` (exit 0), `fail` (exit 1: csfc's diagnostics are kept),
//! `error` (any other exit or a timeout: says nothing about the tree) or
//! `skipped` (with the reason). `emit` is never run: guards do not write.

use super::csfc::{self, Diagnostic};
use super::source::Source;
use super::parent_dir;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const GATES: [&str; 2] = ["check", "check-generated"];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardVerdict {
    /// `csfc check` or `csfc check-generated`.
    pub gate: String,
    /// Tree-relative path of the architecture source, if there is one.
    pub model: Option<String>,
    /// `pass`, `fail`, `error` or `skipped`.
    pub status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reason: String,
    /// csfc's summary line on success.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<Diagnostic>,
    /// Set by `mine` on the commit tree: a guard is required of an agent's
    /// patch only when the task's own reference fix passes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_of_agent: Option<bool>,
}

impl GuardVerdict {
    fn new(gate: &str, model: Option<&str>, status: &str, reason: impl Into<String>) -> Self {
        GuardVerdict { gate: format!("csfc {gate}"), model: model.map(str::to_string), status: status.into(),
                       reason: reason.into(), summary: String::new(), diagnostics: vec![], required_of_agent: None }
    }
}

/// Run every CSF gate on the checkout `tree`. `sources` names architecture
/// sources beyond the convention (tree-relative).
pub fn guard(tree: &Path, csfc_flag: Option<&Path>, grammar_flag: Option<&Path>, sources: &[String])
    -> Result<Vec<GuardVerdict>> {
    let files = Source::dir(tree).files()?;
    let models: Vec<&String> = files.iter()
        .filter(|f| csfc::is_model_source(f) || sources.contains(f)).collect();
    if models.is_empty() {
        return Ok(GATES.iter().map(|g| GuardVerdict::new(g, None, "skipped",
            format!("no architecture source (<root>/{}) in the tree", csfc::MODEL_SOURCE))).collect());
    }
    let info = csfc::locate(csfc_flag);
    let mut out = Vec::new();
    for m in models {
        let root = csfc::model_root(m, &files);
        let rel = m.strip_prefix(&root).map(|s| s.trim_start_matches('/')).unwrap_or(m).to_string();
        let Some(bin) = info.binary() else {
            out.extend(GATES.iter().map(|g| GuardVerdict::new(g, Some(m), "skipped", info.detail.clone())));
            continue;
        };
        let Some(grammar) = csfc::grammar(grammar_flag, tree, &root) else {
            out.extend(GATES.iter().map(|g| GuardVerdict::new(g, Some(m), "skipped",
                "no CSF grammar: pass --csf-grammar PATH or set RRSI_CSF_GRAMMAR")));
            continue;
        };
        let g = grammar.to_string_lossy().into_owned();
        let projections = super::join(parent_dir(&rel), "generated");
        for gate in GATES {
            let mut args = vec![gate, "--source", rel.as_str(), "--root", ".", "--grammar", g.as_str()];
            if gate == "check-generated" {
                if !tree.join(&root).join(&projections).is_dir() {
                    out.push(GuardVerdict::new(gate, Some(m), "skipped",
                        format!("no projection directory {}", super::join(&root, &projections))));
                    continue;
                }
                args.extend(["--output", projections.as_str()]);
            }
            let r = csfc::run(&bin, &tree.join(&root), &args)?;
            let mut v = match r.exit {
                Some(0) => GuardVerdict::new(gate, Some(m), "pass", ""),
                Some(1) => GuardVerdict::new(gate, Some(m), "fail", "csfc reported diagnostics"),
                Some(code) => GuardVerdict::new(gate, Some(m), "error", format!("csfc exited {code}")),
                None => GuardVerdict::new(gate, Some(m), "error",
                    format!("csfc timed out after {}s", csfc::CSFC_TIMEOUT.as_secs())),
            };
            v.summary = r.stdout.trim().to_string();
            if v.status != "pass" {
                v.diagnostics = csfc::parse_diagnostics(&r.stderr);
            }
            out.push(v);
        }
    }
    Ok(out)
}

/// Mark which guards bind an agent: those the reference fix passed.
pub fn mark_required(verdicts: &mut [GuardVerdict]) {
    for v in verdicts {
        v.required_of_agent = Some(v.status == "pass");
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::csf::csfc::tests::fake_csfc;

    fn tree(files: &[&str]) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        for f in files {
            let p = d.path().join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "x").unwrap();
        }
        d
    }

    const MODEL: &str = "svc/csf/architecture/architecture.csf";
    const GRAMMAR: &str = "svc/csf/compiler/architecture/language.ebnf";

    #[test]
    fn no_model_skips_both_gates() {
        let t = tree(&["main.go"]);
        let v = guard(t.path(), None, None, &[]).unwrap();
        assert_eq!(v.iter().map(|v| v.status.as_str()).collect::<Vec<_>>(), ["skipped", "skipped"]);
        assert!(v[0].reason.contains("no architecture source"));
    }

    #[test]
    fn a_missing_csfc_or_grammar_skips_with_the_reason() {
        let t = tree(&[MODEL]);
        let v = guard(t.path(), Some(&t.path().join("no-csfc")), None, &[]).unwrap();
        assert!(v.iter().all(|v| v.status == "skipped" && v.reason.starts_with("cannot run")), "{v:?}");
        let bin = tempfile::tempdir().unwrap();
        let csfc = fake_csfc(bin.path(), "exit 0");
        if std::env::var_os("RRSI_CSF_GRAMMAR").is_none() {
            let v = guard(t.path(), Some(&csfc), None, &[]).unwrap();
            assert!(v.iter().all(|v| v.status == "skipped" && v.reason.contains("no CSF grammar")), "{v:?}");
        }
    }

    #[test]
    fn pass_fail_error_and_a_missing_projection_directory() {
        let t = tree(&[MODEL, GRAMMAR, "svc/csf/architecture/generated/review_cgen.md"]);
        let bin = tempfile::tempdir().unwrap();
        // check passes; check-generated reports drift; both run in svc/.
        let csfc = fake_csfc(bin.path(), r#"[ "$1" = --help=plain ] && exit 0
[ "$(basename "$PWD")" = svc ] || { echo "wrong cwd $PWD" >&2; exit 9; }
case "$1" in
  check) echo "architecture=shop mode=check declarations=checked source=checked obligations=6";;
  check-generated) echo "csf/architecture/generated/review_cgen.md:1:1: CSF_GENERATED_DRIFT: run csfc emit" >&2; exit 1;;
esac"#);
        let v = guard(t.path(), Some(&csfc), None, &[]).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!((v[0].gate.as_str(), v[0].status.as_str()), ("csfc check", "pass"));
        assert!(v[0].summary.contains("obligations=6") && v[0].diagnostics.is_empty());
        assert_eq!((v[1].gate.as_str(), v[1].status.as_str()), ("csfc check-generated", "fail"));
        assert_eq!(v[1].diagnostics[0].code, "CSF_GENERATED_DRIFT");
        assert_eq!(v[1].model.as_deref(), Some(MODEL));

        let crash = fake_csfc(bin.path(), r#"[ "$1" = --help=plain ] && exit 0; echo boom >&2; exit 2"#);
        let v = guard(t.path(), Some(&crash), None, &[]).unwrap();
        assert!(v.iter().all(|v| v.status == "error" && v.reason == "csfc exited 2"), "{v:?}");
        assert_eq!(v[0].diagnostics[0].code, "UNPARSED");

        let bare = tree(&[MODEL, GRAMMAR]);
        let ok = fake_csfc(bin.path(), "exit 0");
        let v = guard(bare.path(), Some(&ok), None, &[]).unwrap();
        assert_eq!((v[0].status.as_str(), v[1].status.as_str()), ("pass", "skipped"));
        assert!(v[1].reason.contains("no projection directory svc/csf/architecture/generated"), "{}", v[1].reason);
    }

    #[test]
    fn only_passed_guards_bind_the_agent() {
        let mut v = vec![GuardVerdict::new("check", None, "pass", ""), GuardVerdict::new("check", None, "fail", ""),
                         GuardVerdict::new("check", None, "skipped", ""), GuardVerdict::new("check", None, "error", "")];
        mark_required(&mut v);
        assert_eq!(v.iter().map(|v| v.required_of_agent).collect::<Vec<_>>(),
                   [Some(true), Some(false), Some(false), Some(false)]);
    }
}
