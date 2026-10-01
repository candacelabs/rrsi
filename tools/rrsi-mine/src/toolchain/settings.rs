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

use super::{bazel::Bazel, cpp::Cpp, go::Go, python::Python, Sandbox, Toolchains};
use clap::Args;

pub const PYTHON_IMAGE: &str = "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:\
85d4cb1afa769a7338e095b927bee941cf5ec92266c7424b3f6c0f2748567248";
pub const CPP_IMAGE: &str = "mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04@sha256:\
d51703c4fcbe93cd889d38005847521d87cca4d304f33423430daf10a384a332";
pub const BAZEL_IMAGE: &str = "gcr.io/bazel-public/bazel:9.2.0@sha256:\
e59bd66f8daf69f02dbfc18dbd72f0ecfe7926bbda95a5c9eb62433d83b8bd02";

/// `--toolchain` of `list` and `mine`.
#[derive(Args, Clone, Debug)]
pub struct ToolchainChoice {
    /// Which toolchain finds and validates candidates: go (the default),
    /// python (pytest), cpp (CMake + CTest), bazel (`bazel test`), or auto
    /// (every toolchain with a project root at HEAD; a commit goes to the
    /// first that claims it, in that order).
    #[arg(long, default_value = "go", value_parser = ["auto", "go", "python", "cpp", "bazel"])]
    pub toolchain: String,
}

/// Container settings of the Python, C++ and Bazel toolchains.
#[derive(Args, Clone, Debug)]
pub struct Settings {
    #[arg(long, env = "RRSI_PYTHON_IMAGE", default_value = PYTHON_IMAGE)]
    pub python_image: String,
    /// Named volume of the Python virtualenvs and uv's caches.
    #[arg(long, env = "RRSI_PYTHON_DEPS", default_value = "rrsi-pydeps")]
    pub python_deps: String,
    #[arg(long, env = "RRSI_CPP_IMAGE", default_value = CPP_IMAGE)]
    pub cpp_image: String,
    /// Extra cmake configure arguments (repeatable), e.g. -DFOO_TESTS=ON.
    #[arg(long = "cmake-arg", allow_hyphen_values = true)]
    pub cmake_args: Vec<String>,
    #[arg(long, env = "RRSI_BAZEL_IMAGE", default_value = BAZEL_IMAGE)]
    pub bazel_image: String,
    /// Named volume of Bazel's output base, repository and disk caches.
    #[arg(long, env = "RRSI_BAZEL_CACHE", default_value = "rrsi-bazelcache")]
    pub bazel_cache: String,
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

impl Settings {
    fn sandbox(&self, image: &str, timeout: u64) -> Sandbox {
        Sandbox { image: image.to_string(), cpus: self.cpus.clone(), memory: self.memory.clone(), timeout,
                  prefetch_timeout: self.prefetch_timeout }
    }

    /// Every toolchain: `go` as configured, the others from these settings
    /// with the same test timeout.
    pub fn toolchains(&self, go: Go) -> Toolchains {
        let t = go.test_timeout;
        Toolchains { all: vec![
            Box::new(go),
            Box::new(Python { sandbox: self.sandbox(&self.python_image, t), deps: self.python_deps.clone() }),
            Box::new(Cpp { sandbox: self.sandbox(&self.cpp_image, t), cmake_args: self.cmake_args.clone() }),
            Box::new(Bazel { sandbox: self.sandbox(&self.bazel_image, t), cache: self.bazel_cache.clone(),
                             lock: Default::default() }),
        ] }
    }
}
