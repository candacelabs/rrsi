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

//! Small-sample intervals for the split comparison: Wilson score intervals
//! for proportions and a seeded percentile bootstrap for medians. Both are
//! 95% intervals; the bootstrap is deterministic (fixed seed) so the report
//! is reproducible.

use super::health::quantile;

/// z for a two-sided 95% interval.
pub const Z95: f64 = 1.959_963_984_540_054;
/// Bootstrap resamples for a median interval.
pub const BOOTSTRAP_RESAMPLES: usize = 4000;
const SEED: u64 = 0x5eed_2026;

/// Wilson score 95% interval of `k` successes in `n` trials, as shares
/// (0..1). `None` when `n` is 0.
pub fn wilson(k: usize, n: usize) -> Option<(f64, f64)> {
    if n == 0 {
        return None;
    }
    let (k, n) = (k as f64, n as f64);
    let p = k / n;
    let z2 = Z95 * Z95;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let half = Z95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    Some(((centre - half).max(0.0), (centre + half).min(1.0)))
}

/// SplitMix64: a tiny deterministic generator for the bootstrap.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Percentile-bootstrap 95% interval of the median of `values`. `None`
/// when there are fewer than two values.
pub fn bootstrap_median_ci(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let mut rng = SplitMix(SEED);
    let n = values.len();
    let mut medians: Vec<f64> = (0..BOOTSTRAP_RESAMPLES).map(|_| {
        let mut s: Vec<f64> = (0..n).map(|_| values[(rng.next() % n as u64) as usize]).collect();
        s.sort_by(f64::total_cmp);
        quantile(&s, 0.5).unwrap_or(0.0)
    }).collect();
    medians.sort_by(f64::total_cmp);
    Some((quantile(&medians, 0.025)?, quantile(&medians, 0.975)?))
}
