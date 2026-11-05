import json
import numpy as np
import matplotlib.pyplot as plt

HISTORY = "/home/qwer1234/sicnav/logs/airport_history.json"
PRED = "/home/qwer1234/sicnav/logs/airport_jmid_predictions.npz"

with open(HISTORY) as f:
    hist = json.load(f)

d = np.load(PRED)

F = d["forecasts"]
ids = d["rmf_ids"]

targets = [12, 18]

idx12 = np.where(ids == 12)[0][0]
idx18 = np.where(ids == 18)[0][0]

past = {}

for target, idx in [(12, idx12), (18, idx18)]:
    arr = []

    for frame in hist["frames"]:
        h = sorted(
            frame["humans"],
            key=lambda x: x["jmid_id"]
        )[idx]

        arr.append([h["x"], h["y"]])

    past[target] = np.array(arr)


plt.figure(figsize=(9, 8))

# past
for target in targets:
    P = past[target]

    plt.plot(
        P[:, 0],
        P[:, 1],
        marker="o",
        linewidth=2,
        label=f"agent{target} past"
    )

    plt.scatter(
        P[-1, 0],
        P[-1, 1],
        marker="x",
        s=100
    )


# 같은 sample index끼리 공동 미래
samples = [0, 3, 7, 11, 14]

for s in samples:

    t12 = F[idx12, s]
    t18 = F[idx18, s]

    line = plt.plot(
        t12[:, 0],
        t12[:, 1],
        alpha=0.6,
        label=f"joint sample {s}"
    )[0]

    plt.plot(
        t18[:, 0],
        t18[:, 1],
        alpha=0.6,
        linestyle="--",
        color=line.get_color()
    )


# robot
robot = hist["frames"][-1]["robot"]

plt.scatter(
    robot["x"],
    robot["y"],
    marker="s",
    s=140,
    label="Robot"
)

plt.axis("equal")
plt.grid(True)

plt.xlabel("world x [m]")
plt.ylabel("world y [m]")

plt.title(
    "JMID joint prediction: agent12 + agent18"
)

plt.legend()

plt.show()
