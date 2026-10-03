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

//! The miner registry. Add a miner: put `src/miners/<module>.rs` (a type
//! implementing [`crate::miner::Miner`]) next to this file and add one
//! `<module> => <Type>` line below. Remove one: delete both.

use crate::miner::Miner;

macro_rules! register {
    ($($module:ident => $miner:ident),* $(,)?) => {
        $(pub mod $module;)*
        /// Every registered miner, in registration order.
        pub fn registry() -> &'static [&'static dyn Miner] {
            &[$(&$module::$miner),*]
        }
    };
}

register! {
    git_history => GitHistory,
    traces => Traces,
    handoffs => Handoffs,
    slices => Slices,
    pr_gap => PrGap,
    copilot_transcripts => CopilotTranscripts,
}

pub fn find(name: &str) -> Option<&'static dyn Miner> {
    registry().iter().copied().find(|m| m.name() == name)
}
