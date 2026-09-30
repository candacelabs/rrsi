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

//! One-shot completions through a locally logged-in agent CLI (GitHub
//! Copilot), mirroring `rrsi/cli_llm.py`: a fresh, tool-less process in an
//! empty scratch directory with the prompt on STDIN, never in argv (one argv
//! string is capped at 128 KiB, and a prompt carries whole patches).
//!
//! The process runner is a trait so tests can check the exact argv and stdin
//! without calling the real CLI.

use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// What one finished process reported.
pub struct Ran {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Runs `argv` in `cwd` with `stdin` as its standard input.
pub trait Runner: Sync {
    fn run(&self, argv: &[String], stdin: &str, cwd: &Path, timeout: Duration) -> Result<Ran>;
}

/// The real runner: a child process, killed at the timeout.
pub struct ProcessRunner;

impl Runner for ProcessRunner {
    fn run(&self, argv: &[String], stdin: &str, cwd: &Path, timeout: Duration) -> Result<Ran> {
        let (prog, args) = argv.split_first().context("empty argv")?;
        let mut child = Command::new(prog).args(args).current_dir(cwd)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().with_context(|| format!("starting {prog}"))?;
        let mut input = child.stdin.take().context("stdin")?;
        let text = stdin.to_string();
        let writer = std::thread::spawn(move || input.write_all(text.as_bytes()));
        let pipe = |mut r: Box<dyn Read + Send>| std::thread::spawn(move || {
            let mut s = String::new();
            let _ = r.read_to_string(&mut s);
            s
        });
        let out = pipe(Box::new(child.stdout.take().context("stdout")?));
        let err = pipe(Box::new(child.stderr.take().context("stderr")?));
        let t0 = Instant::now();
        let status = loop {
            if let Some(s) = child.try_wait()? {
                break s;
            }
            if t0.elapsed() > timeout {
                let _ = child.kill();
                let _ = child.wait();
                bail!("{prog} timed out after {}s", timeout.as_secs());
            }
            std::thread::sleep(Duration::from_millis(200));
        };
        let _ = writer.join();
        Ok(Ran {
            code: status.code().unwrap_or(-1),
            stdout: out.join().unwrap_or_default(),
            stderr: err.join().unwrap_or_default(),
        })
    }
}

/// The Copilot CLI invocation for `model`: silent, no colour, no tools, no
/// built-in MCP servers. The prompt is NOT part of it.
pub fn copilot_argv(model: &str, reasoning: &str) -> Vec<String> {
    ["copilot", "-s", "--no-color", "--available-tools", "", "--disable-builtin-mcps",
     "--model", model, "--reasoning-effort", reasoning]
        .iter().map(|s| s.to_string()).collect()
}

/// A model reached through the Copilot CLI.
pub struct Copilot<'a> {
    pub runner: &'a dyn Runner,
    pub model: String,
    pub reasoning: String,
    pub timeout: Duration,
}

impl Copilot<'_> {
    /// One stateless completion of `prompt`, run in a fresh empty directory.
    pub fn complete(&self, prompt: &str) -> Result<String> {
        let scratch = tempfile::Builder::new().prefix("rrsi-copilot-").tempdir()?;
        let ran = self.runner.run(&copilot_argv(&self.model, &self.reasoning), prompt,
                                  scratch.path(), self.timeout)?;
        let reply = ran.stdout.trim().to_string();
        if ran.code != 0 || reply.is_empty() {
            let why = if ran.stderr.trim().is_empty() { &ran.stdout } else { &ran.stderr };
            let tail: String = why.chars().rev().take(600).collect::<Vec<_>>().into_iter().rev().collect();
            bail!("copilot ({}) rc={}: {}", self.model, ran.code, tail.trim());
        }
        Ok(reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Records every call and answers with a canned reply.
    pub struct Recorder {
        pub calls: Mutex<Vec<(Vec<String>, String)>>,
        pub reply: String,
        pub code: i32,
    }

    impl Runner for Recorder {
        fn run(&self, argv: &[String], stdin: &str, cwd: &Path, _t: Duration) -> Result<Ran> {
            assert!(cwd.is_dir() && std::fs::read_dir(cwd)?.next().is_none(), "cwd must be empty");
            self.calls.lock().unwrap().push((argv.to_vec(), stdin.to_string()));
            Ok(Ran { code: self.code, stdout: self.reply.clone(), stderr: "boom".into() })
        }
    }

    #[test]
    fn the_prompt_goes_on_stdin_never_in_argv() {
        let rec = Recorder { calls: Mutex::new(vec![]), reply: " answer \n".into(), code: 0 };
        let big = format!("PROMPT-MARKER {}", "x".repeat(200_000));
        let llm = Copilot { runner: &rec, model: "m-1".into(), reasoning: "low".into(),
                            timeout: Duration::from_secs(5) };
        assert_eq!(llm.complete(&big).unwrap(), "answer");
        let calls = rec.calls.lock().unwrap();
        let (argv, stdin) = &calls[0];
        assert_eq!(stdin, &big);
        assert!(argv.iter().all(|a| !a.contains("PROMPT-MARKER") && a.len() < 64), "{argv:?}");
        assert_eq!(argv[..2], ["copilot".to_string(), "-s".to_string()]);
        let tools = argv.iter().position(|a| a == "--available-tools").unwrap();
        assert_eq!(argv[tools + 1], "", "no tools: text only");
        assert!(argv.contains(&"--disable-builtin-mcps".to_string()));
        assert_eq!(argv[argv.iter().position(|a| a == "--model").unwrap() + 1], "m-1");
    }

    #[test]
    fn a_failed_or_empty_reply_is_an_error() {
        for (code, reply) in [(1, "text"), (0, "  ")] {
            let rec = Recorder { calls: Mutex::new(vec![]), reply: reply.into(), code };
            let llm = Copilot { runner: &rec, model: "m".into(), reasoning: "low".into(),
                                timeout: Duration::from_secs(5) };
            assert!(llm.complete("p").is_err());
        }
    }

    #[test]
    fn the_process_runner_feeds_stdin_and_times_out() {
        let d = tempfile::tempdir().unwrap();
        let cat = ["cat".to_string()];
        let ran = ProcessRunner.run(&cat, "hello", d.path(), Duration::from_secs(10)).unwrap();
        assert_eq!((ran.code, ran.stdout.as_str()), (0, "hello"));
        let sleep = ["sleep".to_string(), "5".to_string()];
        assert!(ProcessRunner.run(&sleep, "", d.path(), Duration::from_millis(300)).is_err());
    }
}
