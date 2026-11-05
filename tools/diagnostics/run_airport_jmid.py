#!/usr/bin/env python3

import configparser
import json
import os
import sys

import numpy as np


REPO = "/home/qwer1234/sicnav/upstream/safe-interactive-crowdnav"
HISTORY = "/home/qwer1234/sicnav/logs/airport_history.json"

ENV_CONFIG = os.path.join(
    REPO,
    "sicnav_diffusion/configs/env.config"
)

MID_CONFIG = os.path.join(
    REPO,
    "sicnav_diffusion/JMID/test_time_configs/mid_jp.yaml"
)


os.chdir(REPO)
sys.path.insert(0, REPO)

from sicnav_diffusion.JMID.mid_sim_wrapper import (
    HumanTrajectoryForecasterSim
)


with open(HISTORY) as f:
    data = json.load(f)


num_humans = data["num_humans"]
frames = data["frames"]
rmf_ids = data["rmf_ids"]


print("==============================")
print("Airport -> JMID")
print("==============================")
print("humans :", num_humans)
print("frames :", len(frames))
print("RMF IDs:", rmf_ids)


# ---------------------------------------
# SICNav env.config 로드
# ---------------------------------------

cfg = configparser.RawConfigParser()
cfg.read(ENV_CONFIG)

# Airport의 실제 crowd 수 사용
cfg.set(
    "sim",
    "human_num",
    str(num_humans)
)


# ---------------------------------------
# JMID 모델 초기화
# ---------------------------------------

print("\nLoading JMID model...")

forecaster = HumanTrajectoryForecasterSim(
    env_config=cfg,
    mid_config_file=MID_CONFIG,
)

print("JMID loaded")


# ---------------------------------------
# history를 wrapper 형식으로 변환
#
# prev_states:
# N humans x T frames x [x,y,time]
#
# prev_robot_states:
# T frames x [x,y,time]
# ---------------------------------------

prev_states = [
    []
    for _ in range(num_humans)
]

prev_robot_states = []


for frame in frames:

    t = float(frame["time"])

    humans = sorted(
        frame["humans"],
        key=lambda h: h["jmid_id"]
    )

    for h in humans:

        idx = h["jmid_id"]

        prev_states[idx].append([
            float(h["x"]),
            float(h["y"]),
            t,
        ])

    robot = frame["robot"]

    prev_robot_states.append([
        float(robot["x"]),
        float(robot["y"]),
        t,
    ])


forecaster.prev_states = prev_states
forecaster.prev_robot_states = prev_robot_states


print("\nHistory shape")
print(
    f"humans = {len(prev_states)}, "
    f"frames = {len(prev_states[0])}"
)

print(
    f"robot frames = "
    f"{len(prev_robot_states)}"
)


# ---------------------------------------
# Prediction
# ---------------------------------------

print("\nRunning JMID inference...")

forecasts, log_probs = (
    forecaster.predict_ret_best()
)


print("\n==============================")
print("JMID SUCCESS")
print("==============================")

print("forecast shape :", forecasts.shape)
print("log_prob shape :", log_probs.shape)


# expected:
# humans x samples x (current + future) x xy
#
# e.g.
# N x 15 x 9 x 2


out = "/home/qwer1234/sicnav/logs/airport_jmid_predictions.npz"

np.savez(
    out,
    forecasts=forecasts,
    log_probs=log_probs,
    rmf_ids=np.array(rmf_ids),
)


print("\nsaved:", out)


# 첫 번째 사람 첫 sample 출력
print(
    "\nExample:"
    f" agent{rmf_ids[0]}, sample 0"
)

print(forecasts[0, 0])
