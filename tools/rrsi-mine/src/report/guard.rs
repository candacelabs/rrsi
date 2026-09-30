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

//! The report contains private source: refuse to write it into a checkout.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// The nearest ancestor of `out` holding a `.git` (directory or file, so
/// worktrees and submodules count), resolving symlinks of the part of the
/// path that exists.
pub fn enclosing_work_tree(out: &Path) -> Result<Option<PathBuf>> {
    let abs = if out.is_absolute() {
        out.to_path_buf()
    } else {
        std::env::current_dir().context("reading the current directory")?.join(out)
    };
    let mut existing = abs.parent().map(Path::to_path_buf).unwrap_or_default();
    while !existing.exists() {
        match existing.parent() {
            Some(p) => existing = p.to_path_buf(),
            None => break,
        }
    }
    let real = existing.canonicalize().unwrap_or(existing);
    Ok(real.ancestors().find(|a| a.join(".git").exists()).map(Path::to_path_buf))
}
