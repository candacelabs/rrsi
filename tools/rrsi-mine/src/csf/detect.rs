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

//! Is a tree CSF-instrumented, and why.
//!
//! CSF defines no single marker, so detection reports every signal it found:
//!
//! | signal | fires when |
//! |---|---|
//! | `go_mod_is_csf` | a go.mod declares `module github.com/candacelabs/csf` (CSF itself) |
//! | `go_mod_requires_csf` | a go.mod requires `github.com/candacelabs/csf` |
//! | `bazel_module_csf` | a MODULE.bazel call names a module or repository `csf`, e.g. `bazel_dep(name = "csf")` |
//! | `csf_file` | a `*.csf` file is tracked (not necessarily an architecture model) |
//! | `architecture_model` | csfc checked an architecture source and printed its model |
//!
//! Architecture sources are `<root>/csf/architecture/architecture.csf`
//! (csfc's default `--source`) plus any `--csf-source`; other `.csf` files
//! (CSF's documentation vocabulary, for one) are listed with status `other`.
//!
//! go.mod files are read by `gomod-parser` and MODULE.bazel files by
//! `starlark_syntax` (the Starlark parser Bazel's language is defined by),
//! never by string matching. A tree is instrumented when any signal fires.

use super::csfc::{is_model_source, model_root, CsfFile, CsfcInfo, MODEL_SOURCE};
use super::model::LocatedModel;
use super::source::Source;
use super::parent_dir;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use starlark_syntax::dialect::Dialect;
use starlark_syntax::syntax::ast::{ArgumentP, AstLiteral, AstStmt, ExprP};
use starlark_syntax::syntax::uniplate::Visit;
use starlark_syntax::syntax::AstModule;
use std::str::FromStr;

/// The public Go module path of CSF.
pub const CSF_MODULE: &str = "github.com/candacelabs/csf";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signal {
    pub signal: String,
    pub path: String,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoModule {
    /// Repository-relative directory of the go.mod (`""` at the root).
    pub root: String,
    pub module: Option<String>,
    pub requires_csf: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Detection {
    pub instrumented: bool,
    pub signals: Vec<Signal>,
    pub go_modules: Vec<GoModule>,
    /// Every tracked `*.csf` file and what csfc said about it.
    pub csf_files: Vec<CsfFile>,
    /// The files csfc accepted, as csfc printed them.
    pub models: Vec<LocatedModel>,
    /// Every model's `generated` roots, repository-relative, sorted.
    pub generated_paths: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub csfc: Option<CsfcInfo>,
}

/// go.mod and MODULE.bazel files under `testdata/` or `vendor/` are fixtures
/// or copies, not modules of the tree.
fn fixture(path: &str) -> bool {
    path.split('/').any(|s| s == "testdata" || s == "vendor")
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The `module` path and whether CSF is required, from go.mod text.
pub fn read_go_mod(text: &str) -> Result<(String, bool)> {
    let m = gomod_parser::GoMod::from_str(text).map_err(|e| anyhow::anyhow!("parsing go.mod: {e}"))?;
    let requires = m.require.iter().any(|r| r.module.module_path == CSF_MODULE);
    Ok((m.module, requires))
}

/// The callees of MODULE.bazel calls that pass `name = "csf"`, e.g.
/// `["bazel_dep"]`, in source order.
pub fn bazel_csf_calls(path: &str, text: &str) -> Result<Vec<String>> {
    let module = AstModule::parse(path, text.to_string(), &Dialect::Extended)
        .map_err(|e| anyhow::anyhow!("parsing {path}: {e}"))?;
    let mut calls = Vec::new();
    fn visit(v: Visit<'_, starlark_syntax::syntax::ast::AstNoPayload>, calls: &mut Vec<String>) {
        match v {
            Visit::Stmt(s) => s.node.visit_children(|c| visit(c, calls)),
            Visit::Expr(e) => {
                if let ExprP::Call(callee, args) = &e.node {
                    let names_csf = args.args.iter().any(|a| matches!(&a.node,
                        ArgumentP::Named(n, v) if n.node == "name"
                            && matches!(&v.node, ExprP::Literal(AstLiteral::String(s)) if s.node == "csf")));
                    if names_csf {
                        calls.push(callee.to_string());
                    }
                }
                e.node.visit_expr(|c| visit(Visit::Expr(c), calls));
            }
        }
    }
    let stmt: &AstStmt = module.statement();
    visit(Visit::Stmt(stmt), &mut calls);
    Ok(calls)
}

/// Detect CSF instrumentation from the files of `source` alone (no csfc).
/// Architecture sources (see the module docs, plus `extra_sources`) start
/// `unchecked` and [`with_csfc`] adds csfc's verdicts; other `.csf` files
/// are `other`.
pub fn detect(source: &Source, extra_sources: &[String]) -> Result<(Detection, Vec<String>)> {
    let files = source.files()?;
    let mut d = Detection::default();
    for f in &files {
        let name = file_name(f);
        if name == "go.mod" && !fixture(f) {
            let text = source.read_string(f)?.unwrap_or_default();
            let mut m = GoModule { root: parent_dir(f).to_string(), module: None, requires_csf: false, error: None };
            match read_go_mod(&text) {
                Ok((module, requires)) => {
                    if module == CSF_MODULE {
                        d.signals.push(Signal { signal: "go_mod_is_csf".into(), path: f.clone(),
                                                detail: format!("module {CSF_MODULE}") });
                    }
                    if requires {
                        d.signals.push(Signal { signal: "go_mod_requires_csf".into(), path: f.clone(),
                                                detail: format!("require {CSF_MODULE}") });
                    }
                    m.module = Some(module);
                    m.requires_csf = requires;
                }
                Err(e) => m.error = Some(format!("{e:#}")),
            }
            d.go_modules.push(m);
        } else if name == "MODULE.bazel" && !fixture(f) {
            let text = source.read_string(f)?.unwrap_or_default();
            // An unparseable MODULE.bazel names nothing; Bazel would reject it too.
            for call in bazel_csf_calls(f, &text).unwrap_or_default() {
                d.signals.push(Signal { signal: "bazel_module_csf".into(), path: f.clone(),
                                        detail: format!("{call}(name = \"csf\")") });
            }
        } else if name.ends_with(".csf") {
            d.signals.push(Signal { signal: "csf_file".into(), path: f.clone(), detail: String::new() });
            let model = is_model_source(f) || extra_sources.contains(f);
            d.csf_files.push(CsfFile {
                path: f.clone(), root: model_root(f, &files),
                status: if model { "unchecked" } else { "other" }.into(),
                detail: if model { String::new() } else {
                    format!("not at <root>/{MODEL_SOURCE}; pass --csf-source to check it as an architecture")
                },
                diagnostics: vec![],
            });
        }
    }
    d.instrumented = !d.signals.is_empty();
    Ok((d, files))
}

/// Record csfc's verdict on every `.csf` file of `d`, given the checkout
/// `tree` holding the same files.
pub fn with_csfc(d: &mut Detection, files: &[String], tree: &std::path::Path, csfc: CsfcInfo,
                 grammar: Option<&std::path::Path>) -> Result<()> {
    let paths: Vec<String> = d.csf_files.iter().filter(|f| f.status != "other").map(|f| f.path.clone()).collect();
    let (statuses, models) = super::csfc::models(csfc.binary().as_deref(), grammar, tree, &paths, files)?;
    let others = d.csf_files.iter().filter(|f| f.status == "other").cloned();
    let mut statuses: Vec<CsfFile> = statuses.into_iter().chain(others).collect();
    statuses.sort_by(|a, b| a.path.cmp(&b.path));
    for m in &models {
        let path = m.path.clone().unwrap_or_default();
        d.signals.push(Signal { signal: "architecture_model".into(), path,
            detail: format!("architecture {} version {}: {} components", m.model.architecture.name,
                            m.model.architecture.version, m.model.components.len()) });
    }
    d.generated_paths = models.iter().flat_map(LocatedModel::generated_paths).collect();
    d.generated_paths.sort();
    d.generated_paths.dedup();
    d.csf_files = statuses;
    d.models = models;
    d.csfc = Some(csfc);
    d.instrumented = !d.signals.is_empty();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_with(files: &[(&str, &str)]) -> (tempfile::TempDir, Source) {
        let d = tempfile::tempdir().unwrap();
        for (p, text) in files {
            let path = d.path().join(p);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        let s = Source::dir(d.path());
        (d, s)
    }

    fn signals(files: &[(&str, &str)]) -> Vec<String> {
        let (_d, s) = tree_with(files);
        let (d, _) = detect(&s, &[]).unwrap();
        assert_eq!(d.instrumented, !d.signals.is_empty());
        d.signals.into_iter().map(|s| s.signal).collect()
    }

    #[test]
    fn no_signal_means_not_instrumented() {
        assert!(signals(&[("go.mod", "module example.invalid/plain\n\nrequire github.com/other/x v1.0.0\n"),
                          ("MODULE.bazel", "module(name = \"plain\")\nbazel_dep(name = \"rules_go\", version = \"1\")\n"),
                          ("main.go", "package main\n")]).is_empty());
        assert!(signals(&[]).is_empty());
    }

    #[test]
    fn each_signal_fires_alone() {
        assert_eq!(signals(&[("go.mod", "module github.com/candacelabs/csf\n\ngo 1.26\n")]), ["go_mod_is_csf"]);
        assert_eq!(signals(&[("svc/go.mod", "module example.invalid/svc\n\nrequire (\n\tgithub.com/x/y v1.0.0\n\
                                            \tgithub.com/candacelabs/csf v0.1.0 // indirect\n)\n")]),
                   ["go_mod_requires_csf"]);
        assert_eq!(signals(&[("go.mod", "module example.invalid/svc\n\nrequire github.com/candacelabs/csf v0.1.0\n")]),
                   ["go_mod_requires_csf"]);
        assert_eq!(signals(&[("MODULE.bazel", "bazel_dep(\n    name = \"csf\",\n    version = \"0.1.0\",\n)\n")]),
                   ["bazel_module_csf"]);
        assert_eq!(signals(&[("arch/architecture.csf", "architecture shop version 1 {}\n")]), ["csf_file"]);
    }

    #[test]
    fn near_misses_do_not_fire() {
        assert!(signals(&[
            ("go.mod", "module example.invalid/a\n\n// require github.com/candacelabs/csf v1.0.0\n\
                        require github.com/candacelabs/csfx v1.0.0\n\
                        replace github.com/candacelabs/csf => ../csf\n"),
            ("MODULE.bazel", "# bazel_dep(name = \"csf\")\nbazel_dep(name = \"csfx\", version = \"1\")\n\
                              x = \"name = \\\"csf\\\"\"\n"),
            ("testdata/go.mod", "module github.com/candacelabs/csf\n"),
            ("vendor/MODULE.bazel", "bazel_dep(name = \"csf\")\n"),
        ]).is_empty());
    }

    #[test]
    fn bazel_calls_are_found_anywhere_in_the_module() {
        let text = "module(name = \"csf\")\nlocal_repository(name = \"csf\", path = \"x\")\n\
                    ext = use_extension(\"//:e.bzl\", \"e\")\next.tag(name = \"csf\")\n";
        assert_eq!(bazel_csf_calls("MODULE.bazel", text).unwrap(), ["module", "local_repository", "ext.tag"]);
        assert!(bazel_csf_calls("MODULE.bazel", "bazel_dep(name = ").is_err());
    }

    #[test]
    fn detection_lists_go_modules_and_csf_files() {
        let (_d, s) = tree_with(&[
            ("go.mod", "module example.invalid/top\n"),
            ("sub/go.mod", "module example.invalid/sub\n\nrequire github.com/candacelabs/csf v0.1.0\n"),
            ("bad/go.mod", "this is not go.mod\n"),
            ("sub/csf/architecture/architecture.csf", "architecture shop version 1 {}\n"),
            ("sub/csf/docs/vocabulary.csf", "term a \"A\" \"B\";\n"),
        ]);
        let (d, files) = detect(&s, &[]).unwrap();
        assert!(d.instrumented);
        assert_eq!(files.len(), 5);
        let roots: Vec<&str> = d.go_modules.iter().map(|m| m.root.as_str()).collect();
        assert_eq!(roots, ["bad", "", "sub"], "in path order");
        assert!(d.go_modules[0].error.is_some());
        assert!(d.go_modules[2].requires_csf);
        assert_eq!((d.csf_files[0].status.as_str(), d.csf_files[0].root.as_str()), ("unchecked", "sub"));
        assert_eq!(d.csf_files[1].status, "other");
        assert!(d.models.is_empty() && d.csfc.is_none());
        let (d, _) = detect(&s, &["sub/csf/docs/vocabulary.csf".to_string()]).unwrap();
        assert_eq!(d.csf_files[1].status, "unchecked", "--csf-source makes any .csf a model source");
    }

    #[cfg(unix)]
    #[test]
    fn csfc_turns_a_csf_file_into_a_model_with_generated_paths() {
        let (dir, s) = tree_with(&[("svc/go.mod", "module example.invalid/svc\n"),
                                   ("svc/csf/architecture/architecture.csf", "x"), ("docs/v.csf", "y")]);
        let (mut d, files) = detect(&s, &[]).unwrap();
        let json = dir.path().join("..").join(format!("{}.json", dir.path().file_name().unwrap().to_string_lossy()));
        std::fs::write(&json, super::super::model::tests::SHOP).unwrap();
        let csfc = super::super::csfc::tests::fake_csfc(dir.path(), &format!("cat {}", json.display()));
        let info = super::super::csfc::locate(Some(&csfc));
        with_csfc(&mut d, &files, dir.path(), info, Some(std::path::Path::new("/unused.ebnf"))).unwrap();
        std::fs::remove_file(&json).unwrap();
        assert_eq!(d.csf_files[0].status, "other", "docs/v.csf is never sent to csfc");
        assert_eq!(d.csf_files[1].status, "architecture_model");
        assert_eq!(d.csf_files[1].root, "svc");
        assert!(d.signals.iter().any(|s| s.signal == "architecture_model"));
        assert_eq!(d.generated_paths, ["svc/csf/architecture/generated", "svc/services/orders/api_cgen.go"]);
    }
}
