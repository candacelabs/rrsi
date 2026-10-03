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

//! GitHub Copilot CLI sessions as Claude Code-shaped transcripts (stub).

pub use crate::transcript::mtime_size;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};

pub const SESSIONS_FILE: &str = "copilot-sessions.ndjson";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: String,
    pub project: String,
    pub repository: String,
    pub cwd: String,
    pub branch: String,
    pub client: String,
    pub copilot_version: String,
    pub start: String,
    pub end: String,
    pub tool_calls_by_utc_hour: BTreeMap<String, usize>,
    pub human_turns_by_utc_hour: BTreeMap<String, usize>,
    pub user_messages: BTreeMap<String, usize>,
    pub subagent_runs: usize,
    pub tool_errors: usize,
    pub denials: usize,
    pub aborts: usize,
    pub hook_failures: usize,
    pub models: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Converted {
    pub record: SessionRecord,
    pub main: Vec<String>,
    pub subagents: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Summary {
    pub sessions: usize,
    pub processed: usize,
    pub skipped_unchanged: usize,
    pub skipped_since: usize,
    pub removed: usize,
    pub main_transcripts: usize,
    pub subagent_transcripts: usize,
    pub tool_calls: usize,
    pub human_turns: usize,
    pub seconds: f64,
}

pub fn project_of(_cwd: &str) -> String {
    String::new()
}

pub fn exit_code(_text: &str) -> Option<i64> {
    None
}

pub fn map_tool(name: &str, args: &Value) -> (String, Value) {
    (name.to_string(), args.clone())
}

pub fn convert<R: BufRead>(_session_id: &str, _client: &str, _r: R) -> Converted {
    Converted::default()
}

pub fn convert_root(_root: &Path, _out: &Path, _since: &str, _jobs: usize) -> Result<Summary> {
    Ok(Summary::default())
}

pub struct CopilotTranscripts;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    out: PathBuf,
}

impl crate::miner::Miner for CopilotTranscripts {
    fn name(&self) -> &'static str {
        "copilot-transcripts"
    }
    fn about(&self) -> &'static str {
        "GitHub Copilot CLI sessions as Claude Code-shaped transcripts (stub)"
    }
    fn inputs(&self) -> &'static [(&'static str, &'static str)] {
        &[("out", "output directory (outside every work tree)")]
    }
    fn records(&self) -> &'static [(&'static str, &'static str)] {
        &[("<project>/<session>.jsonl", "stub")]
    }
    fn run(&self, args: Value) -> Result<Value> {
        let a: Args = crate::miner::parse_args(self.name(), args)?;
        Ok(serde_json::to_value(convert_root(Path::new(""), &a.out, "", 1)?)?)
    }
}

#[cfg(test)]
mod tests;
