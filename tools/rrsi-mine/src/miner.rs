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

//! The miner plugin contract.
//!
//! A miner turns some input (a git history, session transcripts, ...) into
//! plain JSON records under an output directory. It is one file in
//! `src/miners/` implementing [`Miner`], plus one line in the `register!`
//! list in `src/miners/mod.rs`; deleting both removes it. The CLI needs no
//! change: `rrsi-mine <name> --key value ...` passes the flags to
//! [`Miner::run`] as a JSON object, and `rrsi-mine miners` lists every
//! registered miner with its inputs and records.
//!
//! Arguments and results are plain JSON (no Rust types cross the boundary),
//! so any front end — this CLI, a Python stage, a process gateway — can drive
//! a miner without linking against it.

use anyhow::{bail, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{Map, Value};
use std::path::Path;

/// One mining plugin.
pub trait Miner: Sync {
    /// The CLI name: `rrsi-mine <name> ...`.
    fn name(&self) -> &'static str;
    /// One line on what it mines.
    fn about(&self) -> &'static str;
    /// Its arguments: `(key, meaning)`; `out` is the output directory.
    fn inputs(&self) -> &'static [(&'static str, &'static str)];
    /// The records it writes under `out`: `(path, what one record is)`.
    fn records(&self) -> &'static [(&'static str, &'static str)];
    /// Mines with `args` (a JSON object) and returns a JSON summary.
    fn run(&self, args: Value) -> Result<Value>;
}

/// What `rrsi-mine miners` prints for one miner.
#[derive(Debug, Serialize)]
pub struct Describe {
    pub name: &'static str,
    pub about: &'static str,
    pub inputs: Vec<[&'static str; 2]>,
    pub records: Vec<[&'static str; 2]>,
}

pub fn describe(m: &dyn Miner) -> Describe {
    Describe {
        name: m.name(), about: m.about(),
        inputs: m.inputs().iter().map(|(a, b)| [*a, *b]).collect(),
        records: m.records().iter().map(|(a, b)| [*a, *b]).collect(),
    }
}

/// `--key value`, `--key=value` and bare `--flag` (true) into a JSON object.
/// Hyphens in keys become underscores; a value that parses as a JSON number,
/// boolean, array or object is kept typed; a repeated key becomes an array.
pub fn args_from_cli(argv: &[String]) -> Result<Value> {
    let mut out = Map::new();
    let mut i = 0;
    while i < argv.len() {
        let Some(flag) = argv[i].strip_prefix("--") else { bail!("expected --key, got {:?}", argv[i]) };
        let (key, raw) = match flag.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None if argv.get(i + 1).is_some_and(|n| !n.starts_with("--")) => {
                i += 1;
                (flag.to_string(), Some(argv[i].clone()))
            }
            None => (flag.to_string(), None),
        };
        i += 1;
        let val = match raw {
            None => Value::Bool(true),
            Some(r) => match serde_json::from_str::<Value>(&r) {
                Ok(v @ (Value::Number(_) | Value::Bool(_) | Value::Array(_) | Value::Object(_))) => v,
                _ => Value::String(r),
            },
        };
        let key = key.replace('-', "_");
        match out.get_mut(&key) {
            Some(Value::Array(a)) => a.push(val),
            Some(prev) => *prev = Value::Array(vec![prev.take(), val]),
            None => { out.insert(key, val); }
        }
    }
    Ok(Value::Object(out))
}

/// Typed arguments for a miner; unknown keys are an error.
pub fn parse_args<T: DeserializeOwned>(miner: &str, args: Value) -> Result<T> {
    serde_json::from_value(args).map_err(|e| anyhow::anyhow!("{miner}: {e}"))
}

/// serde helper: accept one value or a list for a `Vec` field.
pub fn one_or_many<'de, D, T>(d: D) -> std::result::Result<Vec<T>, D::Error>
where D: serde::Deserializer<'de>, T: DeserializeOwned {
    use serde::Deserialize;
    match Value::deserialize(d)? {
        Value::Array(a) => a.into_iter().map(|v| serde_json::from_value(v).map_err(serde::de::Error::custom)).collect(),
        Value::Null => Ok(vec![]),
        v => Ok(vec![serde_json::from_value(v).map_err(serde::de::Error::custom)?]),
    }
}

/// The privacy rule every miner shares: records quote private inputs (code,
/// transcripts), so `out` must be outside every git work tree.
pub fn ensure_private_out(miner: &str, out: &Path) -> Result<()> {
    if let Some(tree) = crate::enclosing_work_tree(out) {
        bail!("refusing to write {miner} records to {} inside the git work tree {}: \
               they quote private inputs", out.display(), tree.display());
    }
    Ok(())
}

/// Runs the registered miner `name`; an `out` argument is checked first.
pub fn run(name: &str, args: Value) -> Result<Value> {
    let Some(m) = crate::miners::find(name) else {
        let names: Vec<_> = crate::miners::registry().iter().map(|m| m.name()).collect();
        bail!("no miner {name:?}; registered: {}", names.join(", "));
    };
    if let Some(out) = args.get("out").and_then(Value::as_str) {
        ensure_private_out(name, Path::new(out))?;
    }
    m.run(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn argv(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn cli_flags_become_typed_json() {
        let v = args_from_cli(&argv(&["--out", "/tmp/x", "--jobs", "8", "--since", "2026-06-01", "--dry-run",
                                       "--exclude", "a", "--exclude=b", "--empty", ""])).unwrap();
        assert_eq!(v, json!({"out": "/tmp/x", "jobs": 8, "since": "2026-06-01", "dry_run": true,
                             "exclude": ["a", "b"], "empty": ""}));
        assert!(args_from_cli(&argv(&["positional"])).is_err());
    }

    #[test]
    fn one_or_many_accepts_both() {
        #[derive(serde::Deserialize)]
        struct A {
            #[serde(default, deserialize_with = "one_or_many")]
            x: Vec<String>,
        }
        assert_eq!(parse_args::<A>("t", json!({"x": "a"})).unwrap().x, ["a"]);
        assert_eq!(parse_args::<A>("t", json!({"x": ["a", "b"]})).unwrap().x, ["a", "b"]);
        assert!(parse_args::<A>("t", json!({})).unwrap().x.is_empty());
    }

    #[test]
    fn every_registered_miner_is_findable_and_described() {
        let names: Vec<_> = crate::miners::registry().iter().map(|m| m.name()).collect();
        assert_eq!(names, ["git-history", "traces", "handoffs", "slices", "pr-gap", "copilot-transcripts"]);
        for m in crate::miners::registry() {
            let d = describe(*m);
            assert!(d.inputs.iter().any(|[k, _]| *k == "out"), "{} takes --out", d.name);
            assert!(!d.records.is_empty());
            assert!(crate::miners::find(m.name()).is_some());
        }
    }

    #[test]
    fn unknown_miner_and_private_out_are_refused() {
        assert!(format!("{}", run("nope", json!({})).unwrap_err()).contains("registered: git-history, traces, handoffs, slices, pr-gap, copilot-transcripts"));
        let repo = tempfile::tempdir().unwrap();
        std::fs::create_dir(repo.path().join(".git")).unwrap();
        let out = repo.path().join("o");
        let err = run("traces", json!({"out": out.to_str().unwrap()})).unwrap_err();
        assert!(format!("{err}").contains("inside the git work tree"));
        assert!(!out.exists());
    }

    #[test]
    fn unknown_arguments_are_errors() {
        let err = run("traces", json!({"out": "/nonexistent-rrsi-test", "sinse": "x"})).unwrap_err();
        assert!(format!("{err}").contains("sinse"), "{err}");
    }
}
