# Copyright 2026 Candace Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The hidden per-episode physics of the `phys` simulator backend: physics-v1.

Every graded episode gets its own vehicle, drawn from a frozen distribution by
a random generator seeded with HMAC-SHA256(salt, "<task>--t<i>"). The salt is
a secret file outside the repository; the episode id is the same in every
evaluation, so trial i of scenario X meets the same vehicle (and the same
sensing noise) whichever harness is evaluated and however often: comparisons
between harnesses are paired. Nothing here is drawn from a job name, a batch
id or a run id.

Harness code is forbidden to import this module or to read the salt (the
critic rejects both); that is not prevented, since the harness runs in process
as the same user. Grader output carries only the distribution name, the salt
id (a hash of the salt) and a hash of each episode's draw, never the drawn
values. `--explain` on grader/phys_worker.py prints one draw for an operator.

Changing a range, the draw order or the seeding changes every vehicle: that
is a new distribution name (physics-v1.1, ...), never an edit in place.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import re
import stat

from heldout_vehicle import NOMINAL, Physics

DISTRIBUTION = "physics-v1"
EPISODE_ID = re.compile(r"^(straight-\d+)--t(\d+)$")
WHEELBASE_METRES = 2.7
NOMINAL_CORNERING_N_PER_RAD = 90000.0
NOMINAL_YAW_INERTIA_KGM2 = 2500.0
NOMINAL_MASS_KG = 1500.0

# (name, nominal, low, high, reason). One rng.uniform(low, high) per row, in
# this order; a draw at `scale` s is nominal + s x (uniform - nominal).
RANGES = (
    ("mass_kg", 1500.0, 1275.0, 1950.0, "occupants and cargo"),
    ("inertia_factor", 1.0, 0.9, 1.2, "yaw inertia = 2500 x mass/1500 x factor"),
    ("front_axle_metres", 1.2, 1.10, 1.40, "load distribution; rear axle = 2.7 - front"),
    ("cornering_scale", 1.0, 0.6, 1.25, "tyre wear, pressure, wet road; front = 90000 x scale"),
    ("rear_margin", 0.2, 0.05, 0.40, "rear = front x (a/b + margin): always understeer"),
    ("tyre_friction", 0.9, 0.5, 1.0, "wet to dry asphalt"),
    ("steer_gain", 1.0, 0.80, 1.15, "steering ratio and compliance"),
    ("steer_offset_radians", 0.0, -0.004, 0.004, "alignment pull"),
    ("steer_backlash_radians", 0.0, 0.0, 0.004, "rack play"),
    ("steer_rate_inv", 0.0, 1 / 1.5, 1 / 0.4, "steering rate 0.4-1.5 rad/s"),
    ("accel_rate_inv", 0.0, 1 / 10, 1 / 3, "jerk limit 3-10 m/s^3"),
    ("actuator_dead_ms", 0.0, 0.0, 120.0, "actuator dead time, in 10 ms substeps"),
    ("latency_probability", 0.0, 0.4, 0.4, "chance of one tick of sensing latency"),
    ("rolling_resistance", 0.0, 0.002, 0.006, "kept low: a P controller has no integrator"),
    ("drag_area_m2", 0.0, 0.50, 0.70, "C_d x frontal area"),
    ("accel_gain", 1.0, 0.90, 1.10, "powertrain and brake response"),
    ("lateral_noise_metres", 0.0, 0.0, 0.04, "lane-marking sensing noise"),
    ("heading_noise_radians", 0.0, 0.0, 0.004, "heading sensing noise"),
    ("speed_noise_mps", 0.0, 0.0, 0.08, "speed sensing noise"),
    ("lateral_bias_metres", 0.0, -0.05, 0.05, "camera mounting offset"),
    ("heading_bias_radians", 0.0, -0.003, 0.003, "camera yaw misalignment"),
)


def _rng(salt: bytes, episode_id: str) -> random.Random:
    digest = hmac.new(salt, f"rrsi-sim/{DISTRIBUTION}|{episode_id}".encode(), hashlib.sha256).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def draw(salt: bytes, episode_id: str, scale: float = 1.0) -> tuple[Physics, int]:
    """-> (the episode's vehicle, the seed of its sensing noise). scale 0 is
    NOMINAL with the same noise seed; scale 1 is physics-v1."""
    rng = _rng(salt, episode_id)
    v = {name: nominal + scale * (rng.uniform(low, high) - nominal) for name, nominal, low, high, _ in RANGES}
    latency = 1 if rng.random() < v["latency_probability"] else 0
    noise_seed = rng.getrandbits(63)
    if scale == 0:
        return NOMINAL, noise_seed
    a = v["front_axle_metres"]
    b = WHEELBASE_METRES - a
    front = NOMINAL_CORNERING_N_PER_RAD * v["cornering_scale"]
    physics = Physics(
        mass_kg=v["mass_kg"],
        yaw_inertia_kgm2=NOMINAL_YAW_INERTIA_KGM2 * v["mass_kg"] / NOMINAL_MASS_KG * v["inertia_factor"],
        front_axle_metres=a, rear_axle_metres=b,
        front_cornering_n_per_rad=front, rear_cornering_n_per_rad=front * (a / b + v["rear_margin"]),
        tyre_friction=v["tyre_friction"], steer_gain=v["steer_gain"],
        steer_offset_radians=v["steer_offset_radians"], steer_backlash_radians=v["steer_backlash_radians"],
        steer_rate_rad_per_s=math.inf if v["steer_rate_inv"] <= 0 else 1 / v["steer_rate_inv"],
        accel_rate_mps3=math.inf if v["accel_rate_inv"] <= 0 else 1 / v["accel_rate_inv"],
        actuator_dead_substeps=int(round(v["actuator_dead_ms"] / 10)),
        observation_latency_ticks=latency,
        rolling_resistance=v["rolling_resistance"], drag_area_m2=v["drag_area_m2"],
        accel_gain=v["accel_gain"],
        lateral_noise_metres=v["lateral_noise_metres"], heading_noise_radians=v["heading_noise_radians"],
        speed_noise_mps=v["speed_noise_mps"],
        lateral_bias_metres=v["lateral_bias_metres"], heading_bias_radians=v["heading_bias_radians"])
    return physics, noise_seed


def physics_sha256(physics: Physics, noise_seed: int) -> str:
    """What evidence records instead of the draw: binds it without revealing it."""
    text = json.dumps({"physics": physics.to_dict(), "noise_seed": noise_seed}, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def salt_id(salt: bytes) -> str:
    return hashlib.sha256(salt).hexdigest()[:16]


def load_salt(path: str) -> bytes:
    """The salt file: owner-only (no group or other bits), hex, >= 32 bytes."""
    if not path:
        raise SystemExit("no salt file (RRSI_SIM_PHYS_SALT_FILE is unset)")
    try:
        info = os.stat(path)
    except OSError as error:
        raise SystemExit(f"salt file unreadable: {error.strerror}") from error
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise SystemExit("salt file must be a regular file readable only by its owner (chmod 600)")
    with open(path, encoding="ascii", errors="replace") as handle:
        text = handle.read().strip()
    try:
        salt = bytes.fromhex(text)
    except ValueError as error:
        raise SystemExit("salt file must hold hex") from error
    if len(salt) < 32:
        raise SystemExit("salt must be at least 32 bytes (64 hex digits)")
    return salt
