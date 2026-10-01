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

//! Python: pytest of the changed test files.
//!
//! A project root is the nearest directory holding `pyproject.toml`,
//! `setup.py`, `setup.cfg`, `uv.lock`, `pytest.ini`, `tox.ini` or a
//! `requirements*.txt`. The networked prefetch builds one virtualenv per
//! (root, dependency files) in a shared volume with uv: `uv sync --frozen`
//! when the root has a uv.lock, otherwise every `requirements*.txt` plus the
//! project itself; pytest is always added. The offline run is
//! `python -m pytest <test files>` with that environment.

use super::{in_dir_named, rel, sh_quote, Changed, FileKind, Mapper, Run, Sandbox, Toolchain};
use crate::{relative_package, Candidate, Outcome};
use anyhow::Result;
use std::collections::BTreeSet;
use std::path::Path;

pub struct Python {
    pub sandbox: Sandbox,
    /// Named volume holding the virtualenvs, uv's cache and any Python uv
    /// downloads.
    pub deps: String,
}

pub const TEST_DIRS: [&str; 4] = ["tests", "test", "testing", "testdata"];

/// Whether a file name is a pytest test module (pytest's default
/// `python_files = test_*.py *_test.py`).
pub fn is_test_module(path: &str) -> bool {
    let n = super::file_name(path);
    n.ends_with(".py") && (n.starts_with("test_") || n.ends_with("_test.py"))
}

pub fn classify_file(path: &str) -> FileKind {
    let n = super::file_name(path);
    let py = matches!(super::extension(path), "py" | "pyi");
    if !py {
        // Fixtures and data next to the tests travel with the tests.
        return if in_dir_named(path, &TEST_DIRS) { FileKind::Test } else { FileKind::Other };
    }
    if n.ends_with("_pb2.py") || n.ends_with("_pb2_grpc.py") || n.ends_with("_pb2.pyi")
        || path.contains("/gen/") || path.contains("_cgen") {
        FileKind::Generated
    } else if is_test_module(path) || n == "conftest.py" || in_dir_named(path, &TEST_DIRS) {
        FileKind::Test
    } else {
        FileKind::Source
    }
}

pub fn is_root_marker(path: &str) -> bool {
    let n = super::file_name(path);
    matches!(n, "pyproject.toml" | "setup.py" | "setup.cfg" | "uv.lock" | "pytest.ini" | "tox.ini")
        || (n.starts_with("requirements") && n.ends_with(".txt"))
}

struct PyMapper;

impl Mapper for PyMapper {
    fn root(&self, ch: &Changed) -> Option<String> {
        super::single_root(&ch.tests, &ch.tracked, is_root_marker, false)
    }

    /// The changed test modules that still exist at the commit.
    fn units(&self, ch: &Changed, root: &str, _repo: &Path) -> Result<Vec<String>> {
        Ok(ch.tests.iter().filter(|f| is_test_module(f) && ch.tracked.contains(*f))
            .map(|f| relative_package(root, f)).collect())
    }
}

/// The shell that names the dependency environment of the project in the
/// working directory: a hash of its dependency files.
const ENV_KEY: &str = r#"key=$( (pwd; for f in uv.lock pyproject.toml setup.py setup.cfg requirements*.txt; do [ -f "$f" ] && { echo "== $f"; cat "$f"; }; done) | sha256sum | cut -c1-16 )
env=/deps/env/$key
"#;

pub fn prefetch_script() -> String {
    format!(r#"set -u
{ENV_KEY}export UV_CACHE_DIR=/deps/uv-cache UV_PYTHON_INSTALL_DIR=/deps/python UV_LINK_MODE=copy HOME=/tmp
if [ -x "$env/bin/python" ]; then echo "rrsi: environment $key ready"; exit 0; fi
mkdir -p /deps/env
tmp=/deps/env/.tmp-$key-$$
rm -rf "$tmp"
if [ -f uv.lock ] && [ -f pyproject.toml ]; then
  UV_PROJECT_ENVIRONMENT="$tmp" uv sync --frozen --all-extras --all-groups || exit 1
else
  uv venv -q "$tmp" || exit 1
  for f in requirements*.txt; do
    if [ -f "$f" ]; then uv pip install -q --python "$tmp/bin/python" -r "$f" || exit 1; fi
  done
  if [ -f pyproject.toml ] || [ -f setup.py ]; then
    uv pip install -q --python "$tmp/bin/python" -e '.[test,tests,testing,dev]' \
      || uv pip install -q --python "$tmp/bin/python" -e . || exit 1
  fi
fi
uv pip install -q --python "$tmp/bin/python" pytest || exit 1
mv -T "$tmp" "$env" 2>/dev/null || rm -rf "$tmp"
echo "rrsi: environment $key built"
"#)
}

pub fn test_script(units: &[String]) -> String {
    let args: Vec<String> = units.iter().map(|u| sh_quote(u)).collect();
    format!(r#"{ENV_KEY}if [ ! -x "$env/bin/python" ]; then echo "rrsi: dependency environment $key missing"; exit 97; fi
export PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
"$env/bin/python" -m pytest -p no:cacheprovider -q -rfE --color=no {}
"#, args.join(" "))
}

/// Lines that mean the run never reached the code: no network, no
/// environment, a missing pytest plugin or option.
pub const INFRA_MARKERS: [&str; 10] = [
    "rrsi: dependency environment",
    "Temporary failure in name resolution",
    "Name or service not known",
    "Could not resolve host",
    "Network is unreachable",
    "Failed to establish a new connection",
    "No space left on device",
    "ERROR: file or directory not found:",
    "error: unrecognized arguments:",
    "INTERNALERROR",
];

/// The module a line blames for a failed import: `No module named 'a.b'`,
/// `cannot import name 'X' from 'a.b'`, `module 'a.b' has no attribute`.
fn quoted(rest: &str) -> Option<&str> {
    rest.strip_prefix('\'')?.split('\'').next()
}

pub fn blamed_module(line: &str) -> Option<&str> {
    if let Some(i) = line.find("No module named ") {
        return quoted(&line[i + "No module named ".len()..]);
    }
    if line.contains("cannot import name ") {
        let i = line.find(" from '")?;
        return quoted(&line[i + " from ".len()..]);
    }
    if let Some(i) = line.find("AttributeError: module '") {
        let m = quoted(&line[i + "AttributeError: module ".len()..])?;
        return line.contains("has no attribute").then_some(m);
    }
    None
}

fn top(module: &str) -> &str {
    module.split('.').next().unwrap_or(module)
}

/// Classify one pytest run. `own` holds the top-level module names of the
/// project under test (and of the fix's files): a missing module or name of
/// the project's own code is the code failing to build; one of a
/// third-party distribution is infrastructure.
pub fn classify(exit: i32, log: &str, own: &BTreeSet<String>) -> Outcome {
    if exit == 0 {
        return Outcome::Pass;
    }
    if exit == 137 || exit == 124 {
        return Outcome::Timeout;
    }
    let mut own_missing = false;
    for line in log.lines() {
        if let Some(m) = blamed_module(line) {
            if own.contains(top(m)) {
                own_missing = true;
                continue;
            }
            return Outcome::Infra;
        }
        if INFRA_MARKERS.iter().any(|m| line.contains(m)) {
            return Outcome::Infra;
        }
    }
    if own_missing || log.contains("ERROR collecting") {
        return Outcome::BuildFail;
    }
    let failed = log.lines().any(|l| l.starts_with("FAILED ") || l.starts_with("ERROR ")
        || (l.starts_with('=') && (l.contains(" failed") || l.contains(" error"))));
    if failed {
        return Outcome::TestFail;
    }
    Outcome::Infra
}

/// Top-level module names importable from `root` (its modules and
/// packages, those of a `src/` layout, and the directories of the run's
/// test modules), plus every name along the fix's own source paths: a new
/// module the fix adds is the repository's code even where the parent
/// lacks it.
pub fn own_modules(root: &Path, units: &[String], fix_files: &[String]) -> BTreeSet<String> {
    let mut own = BTreeSet::new();
    let mut add_path = |rel: &Path| {
        let parts: Vec<String> = rel.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        if let Some(first) = parts.first() {
            own.insert(first.trim_end_matches(".py").to_string());
        }
        if parts.len() > 1 && parts[0] == "src" {
            own.insert(parts[1].trim_end_matches(".py").to_string());
        }
    };
    for e in ignore::WalkBuilder::new(root).max_depth(Some(3)).build().flatten() {
        if e.path().extension().is_some_and(|x| x == "py") {
            if let Ok(r) = e.path().strip_prefix(root) {
                add_path(r);
            }
        }
    }
    for u in units {
        let dir = root.join(u.trim_start_matches("./")).parent().map(Path::to_path_buf);
        if let Some(d) = dir {
            for e in ignore::WalkBuilder::new(&d).max_depth(Some(1)).build().flatten() {
                if let Some(s) = e.path().file_stem().filter(|_| e.path().extension().is_some_and(|x| x == "py")) {
                    own.insert(s.to_string_lossy().into_owned());
                }
            }
        }
    }
    for f in fix_files {
        for seg in f.split('/') {
            own.insert(seg.trim_end_matches(".py").to_string());
        }
    }
    own.remove("");
    own
}

impl Python {
    fn run<'a>(&'a self, tree: &'a Path, cand: &Candidate, network: bool, script: String) -> Run<'a> {
        Run { sandbox: &self.sandbox, network, tree, mounts: vec![
                  (tree.display().to_string(), "/src".into()), (self.deps.clone(), "/deps".into())],
              env: vec![], workdir: super::src_dir(&cand.module_root), script,
              timeout: if network { self.sandbox.prefetch_timeout } else { self.sandbox.timeout } }
    }
}

impl Toolchain for Python {
    fn name(&self) -> &'static str {
        "python"
    }

    fn is_root_marker(&self, path: &str) -> bool {
        is_root_marker(path)
    }

    fn classify_file(&self, path: &str) -> FileKind {
        classify_file(path)
    }

    fn candidates(&self, repo: &Path, since: &str) -> Result<Vec<Candidate>> {
        super::scan_candidates(self, &PyMapper, repo, since)
    }

    fn prefetch_label(&self) -> &'static str {
        "uv dependency environment"
    }

    fn test_label(&self, units: &[String]) -> String {
        format!("pytest {}", units.join(" "))
    }

    fn prefetch(&self, tree: &Path, cand: &Candidate) -> Result<(bool, String)> {
        super::own_volume(&self.sandbox.image, &self.deps, tree)?;
        let (code, log) = self.run(tree, cand, true, prefetch_script()).exec()?;
        Ok((code == 0, log))
    }

    fn run_tests(&self, tree: &Path, cand: &Candidate) -> Result<(Outcome, String)> {
        let (code, out) = self.run(tree, cand, false, test_script(&cand.packages)).exec()?;
        let log = format!("$ {}\nexit={code}\n{out}", self.test_label(&cand.packages));
        let fix: Vec<String> = cand.src_files.iter().map(|f| rel(&cand.module_root, f).to_string()).collect();
        let own = own_modules(&tree.join(&cand.module_root), &cand.packages, &fix);
        Ok((classify(code, &log, &own), log))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn own(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn python_files_classify_by_pytest_conventions() {
        assert_eq!(classify_file("pkg/core.py"), FileKind::Source);
        assert_eq!(classify_file("pkg/core.pyi"), FileKind::Source);
        assert_eq!(classify_file("pkg/test_core.py"), FileKind::Test);
        assert_eq!(classify_file("pkg/core_test.py"), FileKind::Test);
        assert_eq!(classify_file("tests/helpers.py"), FileKind::Test);
        assert_eq!(classify_file("conftest.py"), FileKind::Test);
        assert_eq!(classify_file("tests/data/case.json"), FileKind::Test, "fixtures travel with tests");
        assert_eq!(classify_file("pkg/api_pb2.py"), FileKind::Generated);
        assert_eq!(classify_file("pkg/api_pb2_grpc.py"), FileKind::Generated);
        assert_eq!(classify_file("README.md"), FileKind::Other);
        assert_eq!(classify_file("pkg/core.go"), FileKind::Other);
        assert!(is_test_module("a/test_x.py") && is_test_module("a/x_test.py"));
        assert!(!is_test_module("tests/helpers.py") && !is_test_module("conftest.py"));
    }

    #[test]
    fn python_roots_are_the_nearest_dependency_or_config_file() {
        for m in ["pyproject.toml", "a/setup.py", "a/setup.cfg", "uv.lock", "requirements.txt",
                  "b/requirements-test.txt", "pytest.ini", "tox.ini"] {
            assert!(is_root_marker(m), "{m}");
        }
        assert!(!is_root_marker("requirements/README.md") && !is_root_marker("a/b.py"));
    }

    #[test]
    fn test_units_are_the_changed_test_modules_that_still_exist() {
        let tracked: BTreeSet<String> = ["proj/pyproject.toml", "proj/pkg/a.py", "proj/tests/test_a.py",
                                         "proj/tests/helpers.py"].map(String::from).into_iter().collect();
        let ch = Changed { sha: "s".into(), src: vec![(3, "proj/pkg/a.py".into())],
                           tests: vec!["proj/tests/test_a.py".into(), "proj/tests/helpers.py".into(),
                                       "proj/tests/test_gone.py".into()], tracked };
        assert_eq!(PyMapper.root(&ch).as_deref(), Some("proj"));
        assert_eq!(PyMapper.units(&ch, "proj", Path::new(".")).unwrap(), ["./tests/test_a.py"]);
    }

    #[test]
    fn the_scripts_prefetch_with_uv_and_test_offline_with_pytest() {
        let p = prefetch_script();
        assert!(p.contains("uv sync --frozen") && p.contains("-r \"$f\"") && p.contains("pytest"));
        let t = test_script(&["./tests/test_a.py".into(), "./it's.py".into()]);
        assert!(t.contains("-m pytest -p no:cacheprovider -q -rfE --color=no './tests/test_a.py' './it'\\''s.py'"));
        assert!(t.contains("exit 97"), "a missing environment is reported, never tested without it");
    }

    // Regression pins: real pytest output shapes.

    /// A test importing a third-party distribution the offline environment
    /// lacks: infrastructure, never a failing test.
    const MISSING_THIRD_PARTY: &str = "$ pytest ./tests/test_render.py\nexit=2\n\
        ==================================== ERRORS ====================================\n\
        ____________________ ERROR collecting tests/test_render.py _____________________\n\
        ImportError while importing test module '/src/tests/test_render.py'.\n\
        Hint: make sure your test modules/packages have valid Python names.\n\
        Traceback:\n\
        /usr/local/lib/python3.12/importlib/__init__.py:90: in import_module\n\
        \x20   return _bootstrap._gcd_import(name[level:], package, level)\n\
        tests/test_render.py:3: in <module>\n\
        \x20   import yaml\n\
        E   ModuleNotFoundError: No module named 'yaml'\n\
        =========================== short test summary info ============================\n\
        ERROR tests/test_render.py\n\
        !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n\
        1 error in 0.12s\n";

    /// The new test imports a function the parent's own package lacks.
    const NEW_NAME_MISSING: &str = "$ pytest ./tests/test_export.py\nexit=2\n\
        ____________________ ERROR collecting tests/test_export.py _____________________\n\
        ImportError while importing test module '/src/tests/test_export.py'.\n\
        tests/test_export.py:5: in <module>\n\
        \x20   from exporter.manifest import load_manifest, validate_roots\n\
        E   ImportError: cannot import name 'validate_roots' from 'exporter.manifest' (/src/exporter/manifest.py)\n\
        =========================== short test summary info ============================\n\
        ERROR tests/test_export.py\n\
        !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n";

    /// The new test imports a module the fix adds.
    const NEW_MODULE_MISSING: &str = "E   ModuleNotFoundError: No module named 'exporter.snapshots'\n\
        ERROR tests/test_snapshots.py\n";

    /// A third-party name that moved between versions: infrastructure.
    const THIRD_PARTY_NAME: &str = "E   ImportError: cannot import name 'TypeAliasType' from 'typing_extensions' \
        (/deps/env/0123/lib/python3.12/site-packages/typing_extensions.py)\n";

    const ASSERTION: &str = "$ pytest ./tests/test_export.py\nexit=1\n\
        F.                                                                       [100%]\n\
        =================================== FAILURES ===================================\n\
        _____________________________ test_rejects_symlink _____________________________\n\
        \x20   def test_rejects_symlink(tmp_path):\n\
        >       assert validate(tmp_path) == [\"symlink\"]\n\
        E       AssertionError: assert [] == ['symlink']\n\
        tests/test_export.py:12: AssertionError\n\
        =========================== short test summary info ============================\n\
        FAILED tests/test_export.py::test_rejects_symlink - AssertionError: assert [] == ['symlink']\n\
        ========================= 1 failed, 1 passed in 0.05s ==========================\n";

    #[test]
    fn a_missing_third_party_module_is_infra() {
        assert_eq!(classify(2, MISSING_THIRD_PARTY, &own(&["exporter", "tests"])), Outcome::Infra);
        assert_eq!(classify(2, THIRD_PARTY_NAME, &own(&["exporter"])), Outcome::Infra);
    }

    #[test]
    fn a_new_symbol_of_the_repository_missing_in_the_parent_is_a_build_failure() {
        let mine = own(&["exporter", "tests"]);
        assert_eq!(classify(2, NEW_NAME_MISSING, &mine), Outcome::BuildFail);
        assert_eq!(classify(2, NEW_MODULE_MISSING, &mine), Outcome::BuildFail);
        assert_eq!(classify(1, "E   AttributeError: module 'exporter.manifest' has no attribute 'ROOTS'\n\
                                FAILED tests/test_e.py::test_roots\n", &mine), Outcome::BuildFail);
        // The same lines blaming a module that is not the project's: infra.
        assert_eq!(classify(2, NEW_NAME_MISSING, &own(&["other"])), Outcome::Infra);
        // Own-module blame never hides a real third-party failure elsewhere.
        assert_eq!(classify(2, &format!("{NEW_NAME_MISSING}{MISSING_THIRD_PARTY}"), &mine), Outcome::Infra);
    }

    #[test]
    fn assertions_pass_and_odd_exits_classify() {
        let mine = own(&["exporter"]);
        assert_eq!(classify(1, ASSERTION, &mine), Outcome::TestFail);
        assert_eq!(classify(0, "2 passed in 0.01s", &mine), Outcome::Pass);
        assert_eq!(classify(137, "", &mine), Outcome::Timeout);
        assert_eq!(classify(97, "rrsi: dependency environment 0123 missing\n", &mine), Outcome::Infra);
        assert_eq!(classify(5, "no tests ran in 0.01s", &mine), Outcome::Infra, "nothing ran: no evidence");
        assert_eq!(classify(4, "ERROR: usage: pytest\npytest: error: unrecognized arguments: --cov\n", &mine),
                   Outcome::Infra);
        assert_eq!(classify(1, "E   socket.gaierror: [Errno -3] Temporary failure in name resolution\n\
                                FAILED tests/test_x.py::test_fetch\n", &mine), Outcome::Infra);
        assert_eq!(classify(2, "ERROR collecting tests/test_x.py\nE   SyntaxError: invalid syntax\n", &mine),
                   Outcome::BuildFail);
    }

    #[test]
    fn own_modules_come_from_the_tree_the_tests_and_the_fix() {
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        for f in ["exporter/__init__.py", "src/lib_pkg/__init__.py", "tool.py", "tests/test_a.py",
                  "tests/helpers.py", "docs/x.md"] {
            std::fs::create_dir_all(r.join(f).parent().unwrap()).unwrap();
            std::fs::write(r.join(f), "").unwrap();
        }
        let o = own_modules(r, &["./tests/test_a.py".into()], &["newpkg/mod.py".into()]);
        for m in ["exporter", "lib_pkg", "tool", "tests", "helpers", "test_a", "newpkg", "mod"] {
            assert!(o.contains(m), "{m} in {o:?}");
        }
        assert!(!o.contains("docs") && !o.contains("yaml"));
    }
}
