import json
import numpy as np
import matplotlib.pyplot as plt


HISTORY = "/home/qwer1234/sicnav/logs/airport_history.json"

PRED = "/home/qwer1234/sicnav/logs/airport_jmid_predictions.npz"


with open(HISTORY) as f:
    history = json.load(f)

pred = np.load(PRED)

F = pred["forecasts"]
rmf_ids = pred["rmf_ids"]


# sample spread가 가장 큰 사람
spreads = []

for i in range(len(rmf_ids)):

    spread = np.max(
        np.std(
            F[i, :, 1:, :],
            axis=0
        )
    )

    spreads.append(spread)


idx = int(np.argmax(spreads))
rmf_id = int(rmf_ids[idx])

print(
    f"visualizing agent{rmf_id}"
)

print(
    f"sample spread = {spreads[idx]:.4f}"
)


# history
past = []

for frame in history["frames"]:
    h = sorted(
        frame["humans"],
        key=lambda x: x["jmid_id"]
    )[idx]

    past.append([
        h["x"],
        h["y"]
    ])

past = np.array(past)


plt.figure(figsize=(8, 8))


# past trajectory
plt.plot(
    past[:, 0],
    past[:, 1],
    marker="o",
    linewidth=2,
    label="Past"
)


# current position
plt.scatter(
    past[-1, 0],
    past[-1, 1],
    s=100,
    marker="x",
    label="Current"
)


# JMID 15 samples
for s in range(F.shape[1]):

    traj = F[idx, s]

    plt.plot(
        traj[:, 0],
        traj[:, 1],
        alpha=0.35
    )


# robot
r = history["frames"][-1]["robot"]

plt.scatter(
    r["x"],
    r["y"],
    s=120,
    marker="s",
    label="Robot"
)


plt.axis("equal")
plt.grid(True)

plt.xlabel("world x [m]")
plt.ylabel("world y [m]")

plt.title(
    f"JMID prediction - agent{rmf_id}"
)

plt.legend()

plt.show()
