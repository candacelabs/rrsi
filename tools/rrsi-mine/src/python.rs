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

//! `rrsi_mine`: Python bindings (pyo3) over the same library the CLI uses.
//!
//! ```text
//! import rrsi_mine
//! rrsi_mine.list_candidates("/path/to/repo", "2026-06-01") -> list[dict]
//! rrsi_mine.export_tree(repo, sha, dest)
//! rrsi_mine.apply_patch(tree, patch) -> str | None   (error text, or None)
//! rrsi_mine.go_test(tree, module_root, packages, image=..., modcache=...,
//!                   buildcache=..., timeout=600) -> (outcome, log)
//! rrsi_mine.decide(parent_outcome, commit_outcome) -> (valid, reason)
//! rrsi_mine.load_exam(tasks_dir) -> list[dict]   (exam-ready tasks: sha, sha12,
//!                   parent, module_root, packages, instruction)
//! rrsi_mine.csf_guard(tree, csfc=None, grammar=None, sources=None) -> list[dict]
//!                   (CSF's gates on a checkout, e.g. one with an agent's patch
//!                   applied: gate, model, status pass|fail|error|skipped,
//!                   reason, summary, diagnostics [{file, line, col, code, message}])
//! ```

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::path::Path;

fn err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

#[pyfunction]
fn list_candidates(py: Python<'_>, repo: &str, since: &str) -> PyResult<PyObject> {
    let cands = py.allow_threads(|| crate::candidates(Path::new(repo), since)).map_err(err)?;
    let json = serde_json::to_string(&cands).map_err(|e| err(e.into()))?;
    let loads = PyModule::import(py, "json")?.getattr("loads")?;
    Ok(loads.call1((json,))?.unbind())
}

#[pyfunction]
fn export_tree(py: Python<'_>, repo: &str, sha: &str, dest: &str) -> PyResult<()> {
    py.allow_threads(|| crate::export_tree(Path::new(repo), sha, Path::new(dest))).map_err(err)
}

#[pyfunction]
fn apply_patch(py: Python<'_>, tree: &str, patch: &str) -> PyResult<Option<String>> {
    py.allow_threads(|| crate::apply_patch(Path::new(tree), patch)).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (tree, module_root, packages, image = "golang:1.26.5",
                    modcache = "rrsi-gomodcache", buildcache = "rrsi-gobuildcache",
                    timeout = 600))]
#[allow(clippy::too_many_arguments)]
fn go_test(py: Python<'_>, tree: &str, module_root: &str, packages: Vec<String>, image: &str,
           modcache: &str, buildcache: &str, timeout: u64) -> PyResult<(String, String)> {
    let docker = crate::Docker { image, modcache, buildcache, test_timeout: timeout };
    let (outcome, log) = py.allow_threads(|| docker.go_test(Path::new(tree), module_root, &packages))
        .map_err(err)?;
    Ok((format!("{outcome:?}"), log))
}

fn outcome(name: &str) -> PyResult<crate::Outcome> {
    use crate::Outcome::*;
    Ok(match name {
        "Pass" => Pass, "TestFail" => TestFail, "BuildFail" => BuildFail,
        "Infra" => Infra, "Timeout" => Timeout,
        _ => return Err(PyRuntimeError::new_err(format!("unknown outcome {name:?}"))),
    })
}

#[pyfunction]
fn decide(parent: &str, commit: &str) -> PyResult<(bool, &'static str)> {
    Ok(crate::decide(outcome(parent)?, outcome(commit)?))
}

/// The exam-ready tasks of a fairness-checked task directory, judged afresh
/// from the stage verdicts: what the agent sees (`instruction`) and what the
/// harness needs to check it (sha, parent, module_root, packages).
#[pyfunction]
fn load_exam(py: Python<'_>, tasks_dir: &str) -> PyResult<PyObject> {
    let exam = py.allow_threads(|| crate::fairness::load_exam(Path::new(tasks_dir))).map_err(err)?;
    let json = serde_json::to_string(&exam).map_err(|e| err(e.into()))?;
    let loads = PyModule::import(py, "json")?.getattr("loads")?;
    Ok(loads.call1((json,))?.unbind())
}

/// CSF's gates (`csfc check`, `csfc check-generated`) on the checkout
/// `tree`, exactly as `rrsi-mine csf guard` runs them. A grader applies the
/// guards a task marks `required_of_agent` to the agent's patched tree.
#[pyfunction]
#[pyo3(signature = (tree, csfc = None, grammar = None, sources = None))]
fn csf_guard(py: Python<'_>, tree: &str, csfc: Option<&str>, grammar: Option<&str>,
             sources: Option<Vec<String>>) -> PyResult<PyObject> {
    let sources = sources.unwrap_or_default();
    let verdicts = py.allow_threads(|| crate::csf::guard::guard(Path::new(tree), csfc.map(Path::new),
                                                                 grammar.map(Path::new), &sources))
        .map_err(err)?;
    let json = serde_json::to_string(&verdicts).map_err(|e| err(e.into()))?;
    let loads = PyModule::import(py, "json")?.getattr("loads")?;
    Ok(loads.call1((json,))?.unbind())
}

#[pymodule]
fn rrsi_mine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(list_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(export_tree, m)?)?;
    m.add_function(wrap_pyfunction!(apply_patch, m)?)?;
    m.add_function(wrap_pyfunction!(go_test, m)?)?;
    m.add_function(wrap_pyfunction!(decide, m)?)?;
    m.add_function(wrap_pyfunction!(load_exam, m)?)?;
    m.add_function(wrap_pyfunction!(csf_guard, m)?)?;
    Ok(())
}
