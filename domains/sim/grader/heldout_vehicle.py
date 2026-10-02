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
"""The grader-side vehicle: a dynamic single-track model with hidden physics.

This is never the rollout preview. The preview is HighwayEnv's kinematic
bicycle (bench/fake_backend.py): exact state, no tyres, no actuator dynamics.
Graded episodes on the `phys` simulator backend run this vehicle instead, with
parameters drawn per episode (grader/heldout_physics.py), so a controller that
passes the preview can still fail the grade, as a controller tuned in a
simulator can fail on a real car. Harness code is forbidden to import this
module (the critic rejects it); it is not prevented, since the harness runs
in process as the same user. See README "Held-out physics".

The model is the textbook dynamic bicycle: static axle loads, linear tyres
saturating at friction x load, longitudinal, lateral and yaw equations
integrated with 10 explicit Euler substeps per 100 ms control tick. On top of
it sit the actuator (steering gain, offset, backlash, rate limit, dead time;
acceleration gain and jerk limit), rolling and aerodynamic resistance, and
what the controller observes: Gaussian noise and a constant bias on lateral
offset and heading, noise on speed, and optionally one tick of latency. The
oracles always grade the true state.

The controller is evaluated by CSF (Candace Labs' Go framework for AI-agent
systems): its bounded controller runtime, or the in-process mirror of it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

G = 9.81          # m/s^2
AIR = 1.2         # air density, kg/m^3
SUBSTEPS = 10     # explicit Euler substeps per control tick


@dataclass(frozen=True)
class Physics:
    """One vehicle. The defaults are NOMINAL; the graded backend never uses
    them as they are (every graded episode gets its own draw)."""

    mass_kg: float = 1500.0
    yaw_inertia_kgm2: float = 2500.0
    front_axle_metres: float = 1.2            # centre of mass to front axle
    rear_axle_metres: float = 1.5             # centre of mass to rear axle
    front_cornering_n_per_rad: float = 90000.0
    rear_cornering_n_per_rad: float = 90000.0
    tyre_friction: float = 0.9                # peak force / normal load
    steer_gain: float = 1.0                   # road-wheel angle per commanded angle
    steer_offset_radians: float = 0.0         # steering pull (misalignment)
    steer_backlash_radians: float = 0.0       # dead band between rack and wheel
    steer_rate_rad_per_s: float = math.inf
    accel_rate_mps3: float = math.inf         # jerk limit
    actuator_dead_substeps: int = 0           # commands act this many 10 ms substeps late
    observation_latency_ticks: int = 0        # the controller sees a state this many ticks old
    rolling_resistance: float = 0.0           # coefficient c_rr
    drag_area_m2: float = 0.0                 # C_d x frontal area
    accel_gain: float = 1.0                   # delivered / commanded acceleration
    lateral_noise_metres: float = 0.0         # standard deviations of the sensing noise
    heading_noise_radians: float = 0.0
    speed_noise_mps: float = 0.0
    lateral_bias_metres: float = 0.0          # constant sensing biases
    heading_bias_radians: float = 0.0

    def to_dict(self) -> dict:
        """JSON-ready; an infinite rate limit is written as None."""
        return {k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in asdict(self).items()}


NOMINAL = Physics()


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class HeldoutVehicle:
    """Road frame: x along the lane, y left of the lane centre, yaw left
    positive. The interface is the one fake_backend.run_episode drives:
    observe() -> what the controller sees, step(steering_rad,
    acceleration_mps2, dt), state() -> the true state the oracles grade."""

    def __init__(self, scenario: dict, physics: Physics, rng):
        self.p, self.rng = physics, rng
        self.x, self.y = 0.0, float(scenario.get("initial_lateral_metres", 0.0))
        self.yaw = float(scenario.get("initial_heading_radians", 0.0))
        self.vx, self.vy, self.r = 0.6 * float(scenario["target_speed_mps"]), 0.0, 0.0
        self.act_steer, self.wheel, self.accel = 0.0, 0.0, 0.0
        self.queue = deque([(0.0, 0.0)] * physics.actuator_dead_substeps)
        self.obs: deque = deque()

    def step(self, steer_cmd: float, accel_cmd: float, dt: float) -> None:
        p, h = self.p, dt / SUBSTEPS
        a, b = p.front_axle_metres, p.rear_axle_metres
        wheelbase = a + b
        load_f, load_r = p.mass_kg * G * b / wheelbase, p.mass_kg * G * a / wheelbase
        want_s = p.steer_gain * steer_cmd + p.steer_offset_radians
        want_a = p.accel_gain * accel_cmd
        half_backlash = p.steer_backlash_radians / 2
        for _ in range(SUBSTEPS):
            # Actuator: dead time, then rate limits, then backlash to the wheel.
            self.queue.append((want_s, want_a))
            ws, wa = self.queue.popleft()
            self.act_steer += _clip(ws - self.act_steer, p.steer_rate_rad_per_s * h)
            self.accel += _clip(wa - self.accel, p.accel_rate_mps3 * h)
            if self.act_steer - self.wheel > half_backlash:
                self.wheel = self.act_steer - half_backlash
            elif self.wheel - self.act_steer > half_backlash:
                self.wheel = self.act_steer + half_backlash
            d = self.wheel
            # Tyres: slip angles, linear forces saturating at friction x load.
            vxe = max(self.vx, 0.5)
            alpha_f = d - math.atan2(self.vy + a * self.r, vxe)
            alpha_r = -math.atan2(self.vy - b * self.r, vxe)
            fyf = _clip(p.front_cornering_n_per_rad * alpha_f, p.tyre_friction * load_f)
            fyr = _clip(p.rear_cornering_n_per_rad * alpha_r, p.tyre_friction * load_r)
            resist = 0.0
            if self.vx > 0.05:
                resist = p.rolling_resistance * G + 0.5 * AIR * p.drag_area_m2 * self.vx * self.vx / p.mass_kg
            ax = _clip(self.accel, p.tyre_friction * G) - resist
            dvx = ax + self.vy * self.r - fyf * math.sin(d) / p.mass_kg
            dvy = (fyf * math.cos(d) + fyr) / p.mass_kg - vxe * self.r
            dr = (a * fyf * math.cos(d) - b * fyr) / p.yaw_inertia_kgm2
            self.vx = max(0.0, self.vx + dvx * h)
            self.vy += dvy * h
            if self.vx < 0.05:
                self.vy *= 0.5
                self.r *= 0.5
            self.r += dr * h
            self.x += (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * h
            self.y += (self.vx * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * h
            self.yaw += self.r * h

    def state(self) -> dict:
        return {"longitudinal_metres": self.x, "lateral_metres": self.y,
                "heading_error_radians": _wrap(self.yaw), "speed_mps": math.hypot(self.vx, self.vy),
                "collisions": 0}

    def observe(self) -> dict:
        """What the controller sees. Exactly three rng.gauss calls per tick,
        whatever the parameters, so every arm meets the same noise sequence."""
        p, s = self.p, self.state()
        s["lateral_metres"] += p.lateral_bias_metres + self.rng.gauss(0.0, p.lateral_noise_metres)
        s["heading_error_radians"] += p.heading_bias_radians + self.rng.gauss(0.0, p.heading_noise_radians)
        s["speed_mps"] += self.rng.gauss(0.0, p.speed_noise_mps)
        self.obs.append(s)
        while len(self.obs) > p.observation_latency_ticks + 1:
            self.obs.popleft()
        return self.obs[0]
