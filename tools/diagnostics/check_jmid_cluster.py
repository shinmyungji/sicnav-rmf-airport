import configparser
import json
import os
import sys

REPO = "/home/qwer1234/sicnav/upstream/safe-interactive-crowdnav"
HISTORY = "/home/qwer1234/sicnav/logs/airport_history.json"

os.chdir(REPO)
sys.path.insert(0, REPO)

from sicnav_diffusion.JMID.mid_sim_wrapper import HumanTrajectoryForecasterSim


with open(HISTORY) as f:
    data = json.load(f)

rmf_ids = data["rmf_ids"]
num_humans = data["num_humans"]


cfg = configparser.RawConfigParser()

cfg.read(
    os.path.join(
        REPO,
        "sicnav_diffusion/configs/env.config"
    )
)

cfg.set(
    "sim",
    "human_num",
    str(num_humans)
)


forecaster = HumanTrajectoryForecasterSim(
    env_config=cfg,
    mid_config_file=os.path.join(
        REPO,
        "sicnav_diffusion/JMID/test_time_configs/mid_jp.yaml"
    ),
)


prev_states = [
    []
    for _ in range(num_humans)
]

prev_robot_states = []


for frame in data["frames"]:

    t = float(frame["time"])

    humans = sorted(
        frame["humans"],
        key=lambda h: h["jmid_id"]
    )

    for h in humans:

        prev_states[h["jmid_id"]].append([
            float(h["x"]),
            float(h["y"]),
            t,
        ])

    r = frame["robot"]

    prev_robot_states.append([
        float(r["x"]),
        float(r["y"]),
        t,
    ])


_, _, cluster, fallback, _ = (
    forecaster.convert_to_mid_state_env(
        prev_states,
        prev_robot_states
    )
)


print("\n=== JMID joint-prediction cluster ===")

for idx in cluster:
    idx = int(idx)

    print(
        f"JMID index {idx}"
        f" -> agent{rmf_ids[idx]}"
    )


print("\n=== Constant-velocity fallback ===")

for idx in fallback:
    idx = int(idx)

    print(
        f"JMID index {idx}"
        f" -> agent{rmf_ids[idx]}"
    )
