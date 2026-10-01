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

//! WIP: mine the "active agent without a PR" gap (kaashmonee/candace-server#294).

pub struct PrGap;

impl crate::miner::Miner for PrGap {
    fn name(&self) -> &'static str { "pr-gap" }
    fn about(&self) -> &'static str { "WIP" }
    fn inputs(&self) -> &'static [(&'static str, &'static str)] { &[("out", "output directory")] }
    fn records(&self) -> &'static [(&'static str, &'static str)] { &[("runs.jsonl", "WIP")] }
    fn run(&self, _args: serde_json::Value) -> anyhow::Result<serde_json::Value> { anyhow::bail!("pr-gap: WIP") }
}
