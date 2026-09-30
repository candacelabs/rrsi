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

//! CSF's architecture model as its compiler prints it.
//!
//! `csfc emit --format json` checks an `architecture.csf` and prints the
//! declarations as one JSON document (`"format": "csf-architecture"`,
//! `"format_version": 1`). The compiler owns the grammar and the checks; this
//! module only deserializes its output, so rrsi-mine never re-implements the
//! CSF language. Fields this crate does not use are still accepted: a newer
//! compiler may add fields without breaking this reader.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const FORMAT: &str = "csf-architecture";
pub const FORMAT_VERSION: u64 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Location {
    pub file: String,
    pub line: u64,
    pub column: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    pub name: String,
    pub version: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verification {
    /// `pending` or `test_reference` (a named test file, not a passed test).
    pub status: String,
    pub test: Option<String>,
}

/// One declared component. `kind` is the role keyword: `service`,
/// `manager`, `library`, `adapter`, `gateway` or `resource`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Component {
    pub name: String,
    pub kind: String,
    pub process: String,
    pub scope: String,
    /// Root-relative file or directory; resources often have none.
    pub source: Option<String>,
    /// `existing` or `planned`.
    pub state: String,
    pub lifecycle: String,
    pub verification: Verification,
    pub at: Location,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootPath {
    pub path: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Obligation {
    pub subject: String,
    pub requirement: String,
    pub evidence: Option<String>,
}

/// The parts of `csfc emit --format json` rrsi-mine reads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Architecture {
    pub format: String,
    pub format_version: u64,
    pub architecture: Identity,
    pub components: Vec<Component>,
    pub scan_roots: Vec<RootPath>,
    pub generated_roots: Vec<RootPath>,
    pub obligations: Vec<Obligation>,
}

impl Architecture {
    /// Parse a `csfc emit --format json` document, refusing other formats and
    /// versions rather than guessing at their meaning.
    pub fn from_json(text: &str) -> Result<Self> {
        let a: Architecture = serde_json::from_str(text).context("parsing csfc JSON")?;
        ensure!(a.format == FORMAT, "not a CSF architecture document: format {:?}", a.format);
        ensure!(a.format_version == FORMAT_VERSION,
                "unsupported csf-architecture format_version {} (this reader knows {FORMAT_VERSION})",
                a.format_version);
        Ok(a)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_json(&text).with_context(|| format!("in {}", path.display()))
    }
}

/// A checked model and the directory its paths are relative to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocatedModel {
    /// Repository-relative path of the `.csf` source, when known.
    pub path: Option<String>,
    /// Repository-relative directory the model's paths are relative to
    /// (csfc's `--root`; `""` is the repository root).
    pub root: String,
    pub model: Architecture,
}

impl LocatedModel {
    /// The model's `generated` roots, repository-relative.
    pub fn generated_paths(&self) -> Vec<String> {
        self.model.generated_roots.iter().map(|g| super::join(&self.root, &g.path)).collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Printed by the real `csfc emit --format json` for the synthetic model
    /// in tests/fixtures/csf/shop.architecture.csf.
    pub const SHOP: &str = include_str!("../../tests/fixtures/csf/shop.architecture.json");

    pub fn shop(root: &str) -> LocatedModel {
        LocatedModel { path: Some(super::super::join(root, "csf/architecture/architecture.csf")),
                       root: root.into(), model: Architecture::from_json(SHOP).unwrap() }
    }

    #[test]
    fn the_compiler_fixture_parses() {
        let a = Architecture::from_json(SHOP).unwrap();
        assert_eq!((a.architecture.name.as_str(), a.architecture.version), ("shop", 1));
        let kinds: Vec<&str> = a.components.iter().map(|c| c.kind.as_str()).collect();
        assert_eq!(kinds, ["manager", "service", "library", "adapter", "gateway", "resource"]);
        assert_eq!(a.components[1].verification.test.as_deref(), Some("services/orders/orders_test.go"));
        assert_eq!(a.components[5].source, None);
        assert_eq!(a.scan_roots.len(), 3);
        assert_eq!(shop("svc").generated_paths(),
                   ["svc/services/orders/api_cgen.go", "svc/csf/architecture/generated"]);
        assert_eq!(a.obligations.len(), 6);
    }

    #[test]
    fn other_formats_and_versions_are_refused() {
        let other = SHOP.replacen("\"csf-architecture\"", "\"something-else\"", 1);
        assert!(format!("{:#}", Architecture::from_json(&other).unwrap_err()).contains("not a CSF architecture"));
        let newer = SHOP.replacen("\"format_version\": 1", "\"format_version\": 2", 1);
        assert!(format!("{:#}", Architecture::from_json(&newer).unwrap_err()).contains("format_version 2"));
        assert!(Architecture::from_json("{").is_err());
    }

    #[test]
    fn unknown_fields_from_a_newer_compiler_are_accepted() {
        let extra = SHOP.replacen("\"format_version\": 1,", "\"format_version\": 1, \"new_field\": [1],", 1);
        assert!(Architecture::from_json(&extra).is_ok());
    }
}
