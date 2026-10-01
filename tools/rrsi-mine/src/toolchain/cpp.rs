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

//! C++ (CMake + CTest): STUBBED. Any use fails fast with [`STUBBED`].
//!
//! The C++ toolchain is stubbed; see
//! https://github.com/kaashmonee/candace-server/issues/273. The last full
//! implementation (CMake file-API and CTestTestfile.cmake mapping of changed
//! test sources to CTest tests, offline configure + build + `ctest -R`) is
//! commit 8f7bede4e32cd023930364cf8e4bdce38ba664ef of candacelabs/rrsi; on
//! fmtlib/fmt it validated 30 of 51 candidates at 354 s per candidate. The
//! improvements it needs before it is worth running (parent-first runs,
//! incremental builds, include-reachability candidates) are recorded on
//! that issue.

use super::{Changed, Config, FileKind, Registration, Toolchain};
use crate::{Candidate, Outcome};
use anyhow::{bail, Result};
use std::path::Path;

pub const STUBBED: &str =
    "C++ toolchain is stubbed; see https://github.com/kaashmonee/candace-server/issues/273";

pub const REGISTRATION: Registration = Registration {
    name: "cpp", language: "C++", fence: "cpp", classify: |_| FileKind::Other, image: "", volumes: &[],
    build: |_: Config| Box::new(Cpp),
};

/// The registered placeholder: refuses every use.
pub struct Cpp;

impl Toolchain for Cpp {
    fn name(&self) -> &'static str {
        "cpp"
    }

    fn check(&self) -> Result<()> {
        bail!(STUBBED)
    }

    fn is_root_marker(&self, _: &str) -> bool {
        false
    }

    fn classify_file(&self, _: &str) -> FileKind {
        FileKind::Other
    }

    fn project_root(&self, _: &Changed) -> Option<String> {
        None
    }

    fn units(&self, _: &Changed, _: &str, _: &dyn Fn(&str) -> Result<String>) -> Result<Vec<String>> {
        bail!(STUBBED)
    }

    fn prefetch_label(&self) -> &'static str {
        "stubbed"
    }

    fn test_label(&self, _: &[String]) -> String {
        STUBBED.to_string()
    }

    fn prefetch(&self, _: &Path, _: &Candidate) -> Result<(bool, String)> {
        bail!(STUBBED)
    }

    fn run_tests(&self, _: &Path, _: &Candidate) -> Result<(Outcome, String)> {
        bail!(STUBBED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stub_refuses_every_use_with_a_pointer_to_the_issue() {
        let tc = Cpp;
        let msg = tc.check().unwrap_err().to_string();
        assert_eq!(msg, "C++ toolchain is stubbed; see https://github.com/kaashmonee/candace-server/issues/273");
        let d = tempfile::tempdir().unwrap();
        let err = crate::history::candidates(d.path(), "2000-01-01", &tc).unwrap_err();
        assert!(err.to_string().contains("C++ toolchain is stubbed"), "{err}");
    }
}
