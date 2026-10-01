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

//! Merged slice PRs as FAIL_TO_PASS exam candidates (work in progress).

use crate::miner::Miner;
use anyhow::{bail, Result};
use serde_json::Value;

pub struct Slices;

impl Miner for Slices {
    fn name(&self) -> &'static str { "slices" }
    fn about(&self) -> &'static str { "FAIL_TO_PASS Go tasks from merged slice PRs, tagged with slice id and ontology signals" }
    fn inputs(&self) -> &'static [(&'static str, &'static str)] { &[("out", "output directory (outside every work tree)")] }
    fn records(&self) -> &'static [(&'static str, &'static str)] { &[("slices.jsonl", "one merged slice PR per line")] }
    fn run(&self, _args: Value) -> Result<Value> { bail!("slices: not implemented yet") }
}
