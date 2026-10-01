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

//! C++: CMake + CTest.
//!
//! The project root is the outermost directory holding a `CMakeLists.txt`
//! (the top-level project owns the CTest tree). The networked prefetch
//! configures the tree into a build directory beside it (FetchContent and
//! friends download then) and asks CMake's file API for the code model; the
//! generated `CTestTestfile.cmake` files list every test and the executable
//! it runs. A changed test file maps to the CTest tests whose executable is
//! built from it. The offline run reconfigures with
//! `FETCHCONTENT_FULLY_DISCONNECTED=ON`, builds only those executables and
//! runs `ctest -R '^(name|...)$'`.

use super::{in_dir_named, rel, sh_quote, Changed, FileKind, Mapper, Run, Sandbox, Toolchain};
use crate::{dir_of, relative_package, Candidate, Outcome};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub struct Cpp {
    pub sandbox: Sandbox,
    /// Extra `cmake` configure arguments, e.g. `-DFOO_BUILD_TESTS=ON`.
    pub cmake_args: Vec<String>,
}

pub const CODE: [&str; 15] =
    ["c", "cc", "cpp", "cxx", "c++", "h", "hh", "hpp", "hxx", "h++", "inl", "ipp", "tpp", "cppm", "ixx"];
pub const TEST_DIRS: [&str; 6] = ["test", "tests", "unittest", "unittests", "testing", "testdata"];

fn is_code(path: &str) -> bool {
    CODE.contains(&super::extension(path))
}

/// A C/C++ file whose name says it is a test (`x_test.cc`, `test_x.cpp`,
/// `x-test.cc`, `x_unittest.cc`).
pub fn is_test_name(path: &str) -> bool {
    let n = super::file_name(path);
    let stem = n.rsplit_once('.').map(|(s, _)| s).unwrap_or(n);
    stem.ends_with("_test") || stem.ends_with("-test") || stem.ends_with("_unittest")
        || stem.ends_with("Test") || stem.starts_with("test_")
}

pub fn classify_file(path: &str) -> FileKind {
    let n = super::file_name(path);
    let cmake = n == "CMakeLists.txt" || super::extension(path) == "cmake";
    let in_tests = in_dir_named(path, &TEST_DIRS);
    if is_code(path) && (n.contains(".pb.") || path.contains("/gen/") || path.contains("_generated")) {
        FileKind::Generated
    } else if in_tests || (is_code(path) && is_test_name(path)) {
        // Everything under a test directory (sources, CMakeLists.txt, data)
        // is part of the tests.
        FileKind::Test
    } else if is_code(path) || cmake {
        FileKind::Source
    } else {
        FileKind::Other
    }
}

pub fn is_root_marker(path: &str) -> bool {
    super::file_name(path) == "CMakeLists.txt"
}

struct CppMapper;

impl Mapper for CppMapper {
    fn root(&self, ch: &Changed) -> Option<String> {
        super::single_root(&ch.tests, &ch.tracked, is_root_marker, true)
    }

    /// The changed test sources that still exist; resolved to CTest tests
    /// on the configured commit tree ([`Toolchain::resolve_units`]).
    fn units(&self, ch: &Changed, root: &str, _repo: &Path) -> Result<Vec<String>> {
        Ok(ch.tests.iter().filter(|f| is_code(f) && ch.tracked.contains(*f)
                && !matches!(super::extension(f), "h" | "hh" | "hpp" | "hxx" | "inl" | "ipp" | "tpp"))
            .map(|f| relative_package(root, f)).collect())
    }
}

/// The build directory of a tree: beside it, never inside the source.
pub fn build_dir(tree: &Path) -> PathBuf {
    let mut name = tree.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".build");
    tree.with_file_name(name)
}

const CONFIGURE: &str = "cmake -S \"$SRC\" -B /build -G Ninja -DCMAKE_BUILD_TYPE=Debug";

pub fn prefetch_script(cmake_args: &[String]) -> String {
    let extra: Vec<String> = cmake_args.iter().map(|a| sh_quote(a)).collect();
    format!("set -u\nmkdir -p /build/.cmake/api/v1/query && touch /build/.cmake/api/v1/query/codemodel-v2\n\
             {CONFIGURE} {} || exit 1\n", extra.join(" "))
}

pub fn test_script(cmake_args: &[String], targets: &[String], tests: &[String]) -> String {
    let extra: Vec<String> = cmake_args.iter().map(|a| sh_quote(a)).collect();
    let targets: Vec<String> = targets.iter().map(|t| sh_quote(t)).collect();
    let names: Vec<String> = tests.iter().map(|t| regex_escape(t)).collect();
    format!("set -u\n\
             {CONFIGURE} -DFETCHCONTENT_FULLY_DISCONNECTED=ON {} || {{ echo 'rrsi: configure failed'; exit 2; }}\n\
             cmake --build /build --target {} || {{ echo 'rrsi: build failed'; exit 3; }}\n\
             ctest --test-dir /build --output-on-failure --no-tests=error -R {}\n",
            extra.join(" "), targets.join(" "), sh_quote(&format!("^({})$", names.join("|"))))
}

/// Escape a CTest name for `ctest -R` (a CMake regular expression).
pub fn regex_escape(s: &str) -> String {
    s.chars().flat_map(|c| {
        let special = "\\^$.|?*+()[]{}".contains(c);
        special.then_some('\\').into_iter().chain(std::iter::once(c))
    }).collect()
}

/// One CTest test: its name and the executable it runs.
#[derive(Clone, Debug, PartialEq)]
pub struct CtestTest {
    pub name: String,
    pub command: Option<String>,
}

/// The tests a generated `CTestTestfile.cmake` declares:
/// `add_test([=[name]=] "/build/bin/exe" args...)`. (`ctest
/// --show-only=json-v1` omits the command of a test whose executable is
/// not built yet, so it cannot map tests before the build.)
pub fn parse_ctest_testfile(text: &str) -> Vec<CtestTest> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("add_test(") else { continue };
        let (name, rest) = if let Some(r) = rest.strip_prefix("[=[") {
            match r.split_once("]=]") { Some(x) => x, None => continue }
        } else {
            match rest.split_once(' ') { Some(x) => x, None => continue }
        };
        let command = rest.trim_start().strip_prefix('"').and_then(|r| r.split('"').next()).map(str::to_string);
        out.push(CtestTest { name: name.to_string(), command });
    }
    out
}

/// Every test declared under the build directory `build`.
pub fn read_ctest(build: &Path) -> Vec<CtestTest> {
    ignore::WalkBuilder::new(build).hidden(false).ignore(false).git_ignore(false).build().flatten()
        .filter(|e| e.file_name() == "CTestTestfile.cmake")
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .flat_map(|t| parse_ctest_testfile(&t)).collect()
}

/// One CMake target from the file API: its name, artifact paths and source
/// paths (relative to the source directory).
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub name: String,
    pub artifacts: Vec<String>,
    pub sources: Vec<String>,
}

/// The targets of the file API's codemodel-v2 reply in `build`.
pub fn read_codemodel(build: &Path) -> Result<Vec<Target>> {
    let reply = build.join(".cmake/api/v1/reply");
    let index = std::fs::read_dir(&reply).with_context(|| format!("reading {}", reply.display()))?
        .flatten().map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("index-")))
        .max().context("no file API index")?;
    let idx: Value = serde_json::from_str(&std::fs::read_to_string(index)?)?;
    let cm = idx["reply"]["codemodel-v2"]["jsonFile"].as_str().context("no codemodel reply")?;
    let model: Value = serde_json::from_str(&std::fs::read_to_string(reply.join(cm))?)?;
    let mut out = Vec::new();
    for cfg in model["configurations"].as_array().into_iter().flatten() {
        for t in cfg["targets"].as_array().into_iter().flatten() {
            let Some(file) = t["jsonFile"].as_str() else { continue };
            let tj: Value = serde_json::from_str(&std::fs::read_to_string(reply.join(file))?)?;
            let paths = |k: &str| -> Vec<String> {
                tj[k].as_array().into_iter().flatten()
                    .filter_map(|x| x["path"].as_str().map(str::to_string)).collect()
            };
            out.push(Target { name: tj["name"].as_str().unwrap_or_default().to_string(),
                              artifacts: paths("artifacts"), sources: paths("sources") });
        }
    }
    Ok(out)
}

/// The CTest tests whose executable is built from one of `test_sources`
/// (paths relative to the project root), with the targets that build them.
pub fn map_tests(targets: &[Target], tests: &[CtestTest], test_sources: &[String])
    -> (Vec<String>, Vec<String>) {
    let mut names = BTreeSet::new();
    let mut used = BTreeSet::new();
    for t in targets.iter().filter(|t| t.sources.iter().any(|s| test_sources.contains(s))) {
        let exes: BTreeSet<&str> = t.artifacts.iter().map(|a| super::file_name(a)).collect();
        for test in tests {
            if test.command.as_deref().is_some_and(|c| exes.contains(super::file_name(c))) {
                names.insert(test.name.clone());
                used.insert(t.name.clone());
            }
        }
    }
    (names.into_iter().collect(), used.into_iter().collect())
}

/// The targets building the executables of the named tests, from a tree's
/// own build directory.
pub fn targets_for(targets: &[Target], tests: &[CtestTest], names: &[String]) -> Vec<String> {
    let exes: BTreeSet<&str> = tests.iter().filter(|t| names.contains(&t.name))
        .filter_map(|t| t.command.as_deref()).map(super::file_name).collect();
    let mut out: BTreeMap<String, ()> = BTreeMap::new();
    for t in targets {
        if t.artifacts.iter().any(|a| exes.contains(super::file_name(a))) {
            out.insert(t.name.clone(), ());
        }
    }
    out.into_keys().collect()
}

/// Lines that mean the run never reached the code.
pub const INFRA_MARKERS: [&str; 10] = [
    "rrsi: no configured build",
    "Could not resolve host",
    "Couldn't resolve host",
    "Failed to download",
    "error: downloading",
    "FETCHCONTENT_FULLY_DISCONNECTED",
    "Could NOT find",
    "No tests were found",
    "No space left on device",
    "Network is unreachable",
];

/// The header a missing-include line names: GCC's `fatal error: X: No such
/// file or directory` or Clang's `fatal error: 'X' file not found`.
pub fn missing_header(line: &str) -> Option<&str> {
    let i = line.find("fatal error: ")?;
    let rest = &line[i + "fatal error: ".len()..];
    rest.split_once(": No such file or directory").map(|(h, _)| h)
        .or_else(|| rest.split_once(" file not found").map(|(h, _)| h))
        .map(|h| h.trim_matches(['\'', '"', '<', '>']))
}

/// Classify one configure + build + ctest run. `own` holds the paths of the
/// project's files (the tree's and the fix's): a missing header or source
/// of the project itself is a build failure; a missing third-party header,
/// package or download is infrastructure.
pub fn classify(exit: i32, log: &str, own: &BTreeSet<String>) -> Outcome {
    if exit == 0 {
        return Outcome::Pass;
    }
    if exit == 137 || exit == 124 {
        return Outcome::Timeout;
    }
    let owns = |h: &str| own.iter().any(|f| f == h || f.ends_with(&format!("/{h}")));
    let mut own_missing = false;
    for line in log.lines() {
        if let Some(h) = missing_header(line) {
            if owns(h) {
                own_missing = true;
                continue;
            }
            return Outcome::Infra;
        }
        if line.contains("Cannot find source file") {
            own_missing = true;
            continue;
        }
        if INFRA_MARKERS.iter().any(|m| line.contains(m)) {
            return Outcome::Infra;
        }
    }
    if own_missing || log.contains("rrsi: build failed") || log.contains("rrsi: configure failed") {
        return Outcome::BuildFail;
    }
    if log.contains("tests failed out of") || log.contains("The following tests FAILED") {
        return Outcome::TestFail;
    }
    Outcome::Infra
}

impl Cpp {
    fn run<'a>(&'a self, tree: &'a Path, cand: &Candidate, network: bool, script: String) -> Result<Run<'a>> {
        let build = build_dir(tree);
        std::fs::create_dir_all(&build)?;
        Ok(Run { sandbox: &self.sandbox, network, tree,
                 mounts: vec![(tree.display().to_string(), "/src".into()), (build.display().to_string(), "/build".into())],
                 env: vec![("SRC".into(), super::src_dir(&cand.module_root)), ("HOME".into(), "/tmp".into())],
                 workdir: "/build".into(), script,
                 timeout: if network { self.sandbox.prefetch_timeout } else { self.sandbox.timeout } })
    }

    fn model(tree: &Path) -> Result<(Vec<Target>, Vec<CtestTest>)> {
        let build = build_dir(tree);
        Ok((read_codemodel(&build)?, read_ctest(&build)))
    }
}

impl Toolchain for Cpp {
    fn name(&self) -> &'static str {
        "cpp"
    }

    fn is_root_marker(&self, path: &str) -> bool {
        is_root_marker(path)
    }

    fn classify_file(&self, path: &str) -> FileKind {
        classify_file(path)
    }

    fn candidates(&self, repo: &Path, since: &str) -> Result<Vec<Candidate>> {
        super::scan_candidates(self, &CppMapper, repo, since)
    }

    fn resolve_units(&self, commit_tree: &Path, cand: &Candidate) -> Result<Vec<String>> {
        let (targets, tests) = Self::model(commit_tree)?;
        let sources: Vec<String> = cand.packages.iter().map(|p| p.trim_start_matches("./").to_string()).collect();
        Ok(map_tests(&targets, &tests, &sources).0)
    }

    fn prefetch_label(&self) -> &'static str {
        "cmake configure"
    }

    fn test_label(&self, units: &[String]) -> String {
        format!("ctest {}", units.join(" "))
    }

    fn prefetch(&self, tree: &Path, cand: &Candidate) -> Result<(bool, String)> {
        let (code, log) = self.run(tree, cand, true, prefetch_script(&self.cmake_args))?.exec()?;
        Ok((code == 0, log))
    }

    fn run_tests(&self, tree: &Path, cand: &Candidate) -> Result<(Outcome, String)> {
        // The targets come from this tree's own configure; when it failed
        // (a test CMakeLists naming a source the parent lacks) build all.
        let targets = match Self::model(tree) {
            Ok((t, tests)) => targets_for(&t, &tests, &cand.packages),
            Err(_) => vec![],
        };
        let targets = if targets.is_empty() { vec!["all".to_string()] } else { targets };
        let script = test_script(&self.cmake_args, &targets, &cand.packages);
        let (code, out) = self.run(tree, cand, false, script)?.exec()?;
        let log = format!("$ {}\nexit={code}\n{out}", self.test_label(&cand.packages));
        let mut own: BTreeSet<String> = super::tracked_like(&tree.join(&cand.module_root));
        own.extend(cand.src_files.iter().map(|f| rel(&cand.module_root, f).to_string()));
        own.extend(cand.src_files.iter().map(|f| dir_of(f)).filter(|d| !d.is_empty()));
        Ok((classify(code, &log, &own), log))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn own(paths: &[&str]) -> BTreeSet<String> {
        paths.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn cpp_files_classify_by_directory_and_name() {
        assert_eq!(classify_file("include/fmt/format.h"), FileKind::Source);
        assert_eq!(classify_file("src/os.cc"), FileKind::Source);
        assert_eq!(classify_file("CMakeLists.txt"), FileKind::Source);
        assert_eq!(classify_file("support/cmake/FindX.cmake"), FileKind::Source);
        assert_eq!(classify_file("test/format-test.cc"), FileKind::Test);
        assert_eq!(classify_file("test/CMakeLists.txt"), FileKind::Test, "test CMake travels with tests");
        assert_eq!(classify_file("test/data/input.txt"), FileKind::Test);
        assert_eq!(classify_file("lib/parser_test.cpp"), FileKind::Test);
        assert_eq!(classify_file("lib/parser_unittest.cc"), FileKind::Test);
        assert_eq!(classify_file("lib/api.pb.cc"), FileKind::Generated);
        assert_eq!(classify_file("README.md"), FileKind::Other);
        assert!(is_root_marker("CMakeLists.txt") && is_root_marker("a/CMakeLists.txt"));
    }

    #[test]
    fn test_units_are_compilable_changed_test_sources() {
        let tracked: BTreeSet<String> = ["CMakeLists.txt", "test/CMakeLists.txt", "test/a-test.cc", "test/util.h",
                                         "src/a.cc"].map(String::from).into_iter().collect();
        let ch = Changed { sha: "s".into(), src: vec![(2, "src/a.cc".into())],
                           tests: vec!["test/a-test.cc".into(), "test/util.h".into(), "test/CMakeLists.txt".into()],
                           tracked };
        assert_eq!(CppMapper.root(&ch).as_deref(), Some(""), "the top-level project");
        assert_eq!(CppMapper.units(&ch, "", Path::new(".")).unwrap(), ["./test/a-test.cc"]);
    }

    #[test]
    fn ctest_tests_map_to_the_executables_built_from_changed_sources() {
        // A real CMake 3.28 CTestTestfile.cmake, before anything is built.
        let file = "# CMake generated Testfile for\n\
            add_test([=[format-test]=] \"/build/bin/format-test\")\n\
            set_tests_properties([=[format-test]=] PROPERTIES  _BACKTRACE_TRIPLES \"/src/test/CMakeLists.txt;40;add_test\")\n\
            add_test([=[os-test]=] \"/build/bin/os-test\" \"--gtest\")\n\
            add_test(lint \"/usr/bin/sh\" \"-c\" \"true\")\n\
            subdirs(\"gtest\")\n";
        let tests = parse_ctest_testfile(file);
        assert_eq!(tests.len(), 3);
        assert_eq!(tests[2], CtestTest { name: "lint".into(), command: Some("/usr/bin/sh".into()) });
        assert_eq!(tests[1], CtestTest { name: "os-test".into(), command: Some("/build/bin/os-test".into()) });
        let targets = vec![
            Target { name: "format-test".into(), artifacts: vec!["bin/format-test".into()],
                     sources: vec!["test/format-test.cc".into(), "test/test-main.cc".into()] },
            Target { name: "os-test".into(), artifacts: vec!["bin/os-test".into()], sources: vec!["test/os-test.cc".into()] },
            Target { name: "fmt".into(), artifacts: vec!["libfmtd.a".into()], sources: vec!["src/format.cc".into()] },
        ];
        assert_eq!(map_tests(&targets, &tests, &["test/format-test.cc".into()]),
                   (vec!["format-test".to_string()], vec!["format-test".to_string()]));
        assert_eq!(map_tests(&targets, &tests, &["test/test-main.cc".into()]).0, ["format-test"]);
        assert!(map_tests(&targets, &tests, &["src/format.cc".into()]).0.is_empty(), "a library is no test");
        assert_eq!(targets_for(&targets, &tests, &["os-test".into()]), ["os-test"]);
    }

    #[test]
    fn the_codemodel_reply_is_read_from_the_file_api() {
        let d = tempfile::tempdir().unwrap();
        let reply = d.path().join(".cmake/api/v1/reply");
        std::fs::create_dir_all(&reply).unwrap();
        std::fs::write(reply.join("index-2026.json"),
                       r#"{"reply":{"codemodel-v2":{"kind":"codemodel","jsonFile":"codemodel-v2-1.json"}}}"#).unwrap();
        std::fs::write(reply.join("codemodel-v2-1.json"),
                       r#"{"configurations":[{"name":"Debug","targets":[{"name":"t","jsonFile":"target-t.json"}]}]}"#).unwrap();
        std::fs::write(reply.join("target-t.json"),
                       r#"{"name":"t","artifacts":[{"path":"bin/t"}],"sources":[{"path":"test/t.cc"}]}"#).unwrap();
        assert_eq!(read_codemodel(d.path()).unwrap(),
                   [Target { name: "t".into(), artifacts: vec!["bin/t".into()], sources: vec!["test/t.cc".into()] }]);
        assert_eq!(build_dir(Path::new("/w/parent")), Path::new("/w/parent.build"));
    }

    #[test]
    fn the_run_reconfigures_offline_and_builds_only_the_mapped_targets() {
        let s = test_script(&["-DFMT_TEST=ON".into()], &["format-test".into()], &["format-test".into(), "a.b".into()]);
        assert!(s.contains("-DFETCHCONTENT_FULLY_DISCONNECTED=ON '-DFMT_TEST=ON'"), "{s}");
        assert!(s.contains("cmake --build /build --target 'format-test'"), "{s}");
        assert!(s.contains("--no-tests=error -R '^(format-test|a\\.b)$'"), "{s}");
        assert!(prefetch_script(&[]).contains("codemodel-v2"));
    }

    // Regression pins: real GCC / CMake / CTest output shapes.

    const THIRD_PARTY_HEADER: &str = "$ ctest json-test\nexit=3\n\
        [1/4] Building CXX object test/CMakeFiles/json-test.dir/json-test.cc.o\n\
        FAILED: test/CMakeFiles/json-test.dir/json-test.cc.o\n\
        /usr/bin/c++ -I/src/include -g -std=gnu++17 -c /src/test/json-test.cc\n\
        /src/test/json-test.cc:9:10: fatal error: gmock/gmock.h: No such file or directory\n\
        \x20   9 | #include <gmock/gmock.h>\n\
        compilation terminated.\n\
        ninja: build stopped: subcommand failed.\n\
        rrsi: build failed\n";

    const OWN_HEADER_MISSING: &str = "$ ctest ranges-test\nexit=3\n\
        FAILED: test/CMakeFiles/ranges-test.dir/ranges-test.cc.o\n\
        /src/test/ranges-test.cc:8:10: fatal error: fmt/ranges.h: No such file or directory\n\
        \x20   8 | #include \"fmt/ranges.h\"\n\
        compilation terminated.\n\
        ninja: build stopped: subcommand failed.\n\
        rrsi: build failed\n";

    const OWN_SYMBOL_MISSING: &str = "$ ctest format-test\nexit=3\n\
        FAILED: test/CMakeFiles/format-test.dir/format-test.cc.o\n\
        /src/test/format-test.cc:2210:16: error: 'format_bytes' is not a member of 'fmt'\n\
        \x202210 |   EXPECT_EQ(fmt::format_bytes(1024), \"1 KiB\");\n\
        ninja: build stopped: subcommand failed.\n\
        rrsi: build failed\n";

    const CTEST_FAILED: &str = "$ ctest format-test\nexit=8\n\
        1/1 Test #3: format-test ......................***Failed    0.05 sec\n\
        [  FAILED  ] FormatTest.Bytes (0 ms)\n\
        0% tests passed, 1 tests failed out of 1\n\
        The following tests FAILED:\n\
        \x20\x20\x20\x20\x20\x20\x203 - format-test (Failed)\n\
        Errors while running CTest\n";

    #[test]
    fn a_missing_third_party_header_or_package_is_infra() {
        let mine = own(&["include/fmt/format.h", "test/json-test.cc"]);
        assert_eq!(classify(3, THIRD_PARTY_HEADER, &mine), Outcome::Infra);
        assert_eq!(classify(2, "CMake Error at CMakeLists.txt:12 (find_package):\n\
                                Could NOT find GTest (missing: GTEST_LIBRARY GTEST_INCLUDE_DIR)\n\
                                rrsi: configure failed\n", &mine), Outcome::Infra);
        assert_eq!(classify(2, "CMake Error: Failed to download https://example.invalid/x.tar.gz\n", &mine),
                   Outcome::Infra);
        assert_eq!(classify(8, "No tests were found!!!\n", &mine), Outcome::Infra);
    }

    #[test]
    fn a_new_header_or_symbol_of_the_repository_missing_in_the_parent_is_a_build_failure() {
        // fmt/ranges.h is a file the fix adds (in `own` via the fix's paths).
        let mine = own(&["include/fmt/format.h", "include/fmt/ranges.h"]);
        assert_eq!(classify(3, OWN_HEADER_MISSING, &mine), Outcome::BuildFail);
        assert_eq!(classify(3, OWN_SYMBOL_MISSING, &mine), Outcome::BuildFail);
        assert_eq!(classify(2, "CMake Error at test/CMakeLists.txt:40 (add_executable):\n\
                                \x20 Cannot find source file:\n    ../src/new.cc\n\
                                rrsi: configure failed\n", &mine), Outcome::BuildFail);
        // The same header line, but not a file of this repository: infra.
        assert_eq!(classify(3, OWN_HEADER_MISSING, &own(&["include/fmt/format.h"])), Outcome::Infra);
        // Own blame never hides a third-party failure elsewhere.
        assert_eq!(classify(3, &format!("{OWN_HEADER_MISSING}{THIRD_PARTY_HEADER}"), &mine), Outcome::Infra);
    }

    #[test]
    fn ctest_failures_passes_and_timeouts_classify() {
        let mine = own(&[]);
        assert_eq!(classify(8, CTEST_FAILED, &mine), Outcome::TestFail);
        assert_eq!(classify(0, "100% tests passed, 0 tests failed out of 1", &mine), Outcome::Pass);
        assert_eq!(classify(137, "", &mine), Outcome::Timeout);
        assert_eq!(classify(1, "something odd", &mine), Outcome::Infra);
        assert_eq!(missing_header("x.cc:1:10: fatal error: 'a/b.h' file not found"), Some("a/b.h"));
        assert_eq!(missing_header("x.cc:1:10: fatal error: a/b.h: No such file or directory"), Some("a/b.h"));
    }
}
