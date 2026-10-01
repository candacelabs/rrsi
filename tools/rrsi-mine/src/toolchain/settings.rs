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

//! Command-line settings of the non-Go toolchains (the Go flags stay where
//! they were in main.rs). Images are pinned by digest; every test run is
//! capped like the Go run.

use super::{Config, Sandbox, Toolchains, REGISTRY};
use clap::builder::PossibleValuesParser;
use clap::Args;

/// `auto` and every registered toolchain name.
fn choices() -> PossibleValuesParser {
    PossibleValuesParser::new(std::iter::once("auto").chain(REGISTRY.iter().map(|r| r.name)))
}

/// `--toolchain` of `list` and `mine`.
#[derive(Args, Clone, Debug)]
pub struct ToolchainChoice {
    /// Which toolchain finds and validates candidates: go (the default),
    /// python (pytest), cpp (CMake + CTest), bazel (`bazel test`), or auto
    /// (every toolchain with a project root at HEAD; a commit goes to the
    /// first that claims it, in that order).
    #[arg(long, default_value = "go", value_parser = choices())]
    pub toolchain: String,
}

/// Container settings of every toolchain but Go (whose flags predate
/// toolchains). Each registered toolchain's pinned image and volumes are
/// the defaults; `NAME=VALUE` flags or `RRSI_<NAME>_IMAGE` override them.
#[derive(Args, Clone, Debug)]
pub struct Settings {
    /// A toolchain's image, e.g. `python=python:3.12@sha256:...` (repeatable;
    /// also RRSI_<NAME>_IMAGE).
    #[arg(long = "toolchain-image", value_name = "NAME=IMAGE")]
    pub images: Vec<String>,
    /// A toolchain's named volumes, comma-separated in its order, e.g.
    /// `bazel=my-bazel-cache` (repeatable).
    #[arg(long = "toolchain-volume", value_name = "NAME=VOLUME[,VOLUME]")]
    pub volumes: Vec<String>,
    /// An extra argument for a toolchain, e.g. `cpp=-DFOO_TESTS=ON` for
    /// cmake configure (repeatable).
    #[arg(long = "toolchain-arg", value_name = "NAME=ARG", allow_hyphen_values = true)]
    pub args: Vec<String>,
    /// CPUs of one test container.
    #[arg(long, env = "RRSI_CPUS", default_value = "4")]
    pub cpus: String,
    /// Memory of one test container.
    #[arg(long, env = "RRSI_MEMORY", default_value = "6g")]
    pub memory: String,
    /// Seconds the networked prefetch may take.
    #[arg(long, env = "RRSI_PREFETCH_TIMEOUT", default_value_t = 3600)]
    pub prefetch_timeout: u64,
}

/// The values of `NAME=VALUE` entries for `name`, in order.
fn values<'a>(entries: &'a [String], name: &str) -> impl Iterator<Item = &'a str> + 'a {
    let prefix = format!("{name}=");
    entries.iter().filter_map(move |e| e.strip_prefix(prefix.as_str()))
}

impl Settings {
    /// The resolved settings of toolchain `name` with test timeout
    /// `timeout`.
    pub fn config(&self, name: &str, timeout: u64) -> Option<Config> {
        let r = super::registration(name)?;
        let env = std::env::var(format!("RRSI_{}_IMAGE", name.to_uppercase())).ok();
        let image = values(&self.images, name).last().map(str::to_string).or(env)
            .unwrap_or_else(|| r.image.to_string());
        let volumes = values(&self.volumes, name).last()
            .map(|v| v.split(',').map(str::to_string).collect())
            .unwrap_or_else(|| r.volumes.iter().map(|v| v.to_string()).collect());
        Some(Config { sandbox: Sandbox { image, cpus: self.cpus.clone(), memory: self.memory.clone(), timeout,
                                         prefetch_timeout: self.prefetch_timeout },
                      volumes, args: values(&self.args, name).map(str::to_string).collect() })
    }

    /// Every registered toolchain; `fixed` overrides a toolchain's whole
    /// config (Go's own flags).
    pub fn toolchains(&self, timeout: u64, fixed: Vec<(&str, Config)>) -> Toolchains {
        let mut fixed = fixed;
        Toolchains { all: REGISTRY.iter().map(|r| {
            let config = match fixed.iter().position(|(n, _)| *n == r.name) {
                Some(i) => fixed.remove(i).1,
                None => self.config(r.name, timeout).expect("registered"),
            };
            (r.build)(config)
        }).collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        s: Settings,
        #[command(flatten)]
        t: ToolchainChoice,
    }

    #[test]
    fn registered_defaults_and_name_value_overrides() {
        let cli = Cli::parse_from(["x", "--toolchain", "cpp", "--toolchain-arg", "cpp=-DFMT_TEST=ON",
                                   "--toolchain-volume", "bazel=a,b", "--toolchain-image", "python=py:1"]);
        assert_eq!(cli.t.toolchain, "cpp");
        assert_eq!(cli.s.config("cpp", 600).unwrap().args, ["-DFMT_TEST=ON"]);
        assert!(cli.s.config("bazel", 600).unwrap().sandbox.image.contains("@sha256:"));
        assert_eq!(cli.s.config("bazel", 1).unwrap().volumes, ["a", "b"]);
        assert_eq!(cli.s.config("python", 1).unwrap().sandbox.image, "py:1");
        assert_eq!(cli.s.config("python", 1).unwrap().volumes, ["rrsi-pydeps"]);
        assert!(cli.s.config("nope", 1).is_none());
        assert!(Cli::try_parse_from(["x", "--toolchain", "rust"]).is_err(), "only registered names");
        let names: Vec<&str> = cli.s.toolchains(600, vec![]).all.iter().map(|t| t.name()).collect();
        assert_eq!(names, ["go", "python", "cpp", "bazel"]);
    }
}
