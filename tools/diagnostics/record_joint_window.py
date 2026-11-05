#!/usr/bin/env python3

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DT = 0.25
NUM_FRAMES = 6
ATTENTION_RADIUS = 3.0

OUT = Path(
    "/home/qwer1234/sicnav/logs/airport_history.json"
)


class JointWindowRecorder(Node):

    def __init__(self):
        super().__init__("joint_window_recorder")

        self.latest_humans = None
        self.latest_robot = None

        self.buffer = deque(maxlen=NUM_FRAMES)

        self.rmf_ids = None
        self.id_mapping = None

        self.create_subscription(
            String,
            "/sicnav/human_states",
            self.human_cb,
            10
        )

        self.create_subscription(
            String,
            "/sicnav/robot_state",
            self.robot_cb,
            10
        )

        self.create_timer(
            DT,
            self.sample
        )

        print()
        print("==============================")
        print("JMID JOINT WINDOW RECORDER")
        print("==============================")
        print(f"attention radius : {ATTENTION_RADIUS} m")
        print(f"history frames   : {NUM_FRAMES}")
        print(f"dt               : {DT} s")
        print()
        print("Waiting until all 3 humans are")
        print("inside the ACTUAL JMID cluster...")
        print()


    def human_cb(self, msg):
        self.latest_humans = json.loads(msg.data)


    def robot_cb(self, msg):
        self.latest_robot = json.loads(msg.data)


    def compute_original_jmid_cluster(
        self,
        humans,
        robot
    ):
        """
        Reproduce the relevant part of:

        mid_sim_wrapper.py
        convert_to_mid_state_env()

        track_id ordering:
          -1 = robot
           0 = agent11
           1 = agent12
           2 = agent13
        """

        humans = sorted(
            humans,
            key=lambda h: h["id"]
        )

        # Stable RMF -> JMID mapping
        if self.rmf_ids is None:

            self.rmf_ids = [
                h["id"]
                for h in humans
            ]

            self.id_mapping = {
                rid: idx
                for idx, rid in enumerate(
                    self.rmf_ids
                )
            }

            print("RMF -> JMID mapping")

            for rid, idx in self.id_mapping.items():
                print(
                    f"  agent{rid} -> {idx}"
                )

            print()


        # EXACT ordering produced after:
        # filtered_df.sort_values(track_id)
        #
        # robot track_id=-1 first.

        positions = [
            [
                robot["x"],
                robot["y"]
            ]
        ]

        track_ids = [-1]

        for h in humans:

            positions.append(
                [
                    h["x"],
                    h["y"]
                ]
            )

            track_ids.append(
                self.id_mapping[h["id"]]
            )


        positions = np.asarray(
            positions,
            dtype=float
        )

        track_ids = np.asarray(
            track_ids,
            dtype=int
        )


        # Same distance matrix as wrapper
        sq_diffs = np.square(
            positions[:, None]
            -
            positions[None, :]
        )

        dists = np.sqrt(
            np.sum(
                sq_diffs,
                axis=2
            )
        )


        # Same:
        # mask = dists < attention_radius
        mask = (
            dists
            <
            ATTENTION_RADIUS
        )


        # Same cluster means
        cluster_sums = (
            mask.astype(float)
            @ positions
        )

        cluster_counts = (
            mask.sum(
                axis=1,
                keepdims=True
            )
        )

        cluster_means = (
            cluster_sums
            /
            cluster_counts
        )


        # Original wrapper:
        #
        # robot_dist =
        # norm(cluster_means - positions[0])
        #
        # chosen_cluster_idx =
        # argmin(robot_dist[1:]) + 1
        #
        # => ignore robot as center candidate

        robot_dist = np.linalg.norm(
            cluster_means
            -
            positions[0],
            axis=1
        )

        chosen_idx = (
            np.argmin(
                robot_dist[1:]
            )
            + 1
        )


        selected_track_ids = (
            track_ids[
                mask[chosen_idx]
            ]
        )

        selected_humans = [
            int(t)
            for t in selected_track_ids
            if t != -1
        ]


        # Convert back to RMF IDs
        reverse_mapping = {
            v: k
            for k, v
            in self.id_mapping.items()
        }

        selected_rmf_ids = [
            reverse_mapping[t]
            for t in selected_humans
        ]


        # useful diagnostics
        chosen_track_id = int(
            track_ids[chosen_idx]
        )

        chosen_rmf_id = (
            reverse_mapping[
                chosen_track_id
            ]
        )


        return {
            "selected_jmid_ids":
                selected_humans,

            "selected_rmf_ids":
                selected_rmf_ids,

            "chosen_center_rmf_id":
                chosen_rmf_id,

            "dist_matrix":
                dists
        }


    def sample(self):

        if (
            self.latest_humans is None
            or
            self.latest_robot is None
        ):
            return


        humans = sorted(
            self.latest_humans,
            key=lambda h: h["id"]
        )

        robot = self.latest_robot


        if len(humans) != 3:

            print(
                f"\rWaiting: humans={len(humans)}",
                end="",
                flush=True
            )

            return


        result = self.compute_original_jmid_cluster(
            humans,
            robot
        )


        # ------------------------------------------------
        # Add current frame to rolling history
        # ------------------------------------------------

        frame = {
            "time": 0.0,

            "humans": [
                {
                    "jmid_id":
                        self.id_mapping[h["id"]],

                    "rmf_id":
                        h["id"],

                    "x":
                        h["x"],

                    "y":
                        h["y"]
                }
                for h in humans
            ],

            "robot": {
                "x": robot["x"],
                "y": robot["y"],
                "yaw": robot.get(
                    "yaw",
                    0.0
                )
            }
        }


        self.buffer.append(frame)


        # ------------------------------------------------
        # diagnostics
        # ------------------------------------------------

        selected = sorted(
            result[
                "selected_rmf_ids"
            ]
        )

        chosen = result[
            "chosen_center_rmf_id"
        ]


        h11 = humans[0]
        h12 = humans[1]
        h13 = humans[2]


        def dist(a, b):

            return math.hypot(
                a["x"] - b["x"],
                a["y"] - b["y"]
            )


        d12 = dist(h11, h12)
        d13 = dist(h11, h13)
        d23 = dist(h12, h13)


        print(
            "\r"
            f"chosen=agent{chosen}  "
            f"JMID={selected}  |  "
            f"d11-12={d12:.2f}  "
            f"d11-13={d13:.2f}  "
            f"d12-13={d23:.2f}  "
            f"buffer={len(self.buffer)}/{NUM_FRAMES}",
            end="",
            flush=True
        )


        # ------------------------------------------------
        # Success:
        #
        # original JMID selection contains all 3 humans
        # AND rolling history already has 6 frames
        # ------------------------------------------------

        expected = sorted(
            self.rmf_ids
        )

        if (
            selected == expected
            and
            len(self.buffer) == NUM_FRAMES
        ):

            print()
            print()
            print("==============================")
            print("3-HUMAN JMID WINDOW FOUND")
            print("==============================")

            print(
                "chosen center : "
                f"agent{chosen}"
            )

            print(
                "JMID cluster  :",
                selected
            )


            # rewrite timestamps as
            # 0, .25, .50 ... 1.25
            frames = list(
                self.buffer
            )

            for i, f in enumerate(
                frames
            ):

                f["time"] = (
                    i * DT
                )


            output = {
                "dt":
                    DT,

                "num_frames":
                    NUM_FRAMES,

                "num_humans":
                    len(self.rmf_ids),

                "rmf_ids":
                    self.rmf_ids,

                "id_mapping": {
                    str(k): v
                    for k, v
                    in self.id_mapping.items()
                },

                "frames":
                    frames
            }


            OUT.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                OUT,
                "w"
            ) as f:

                json.dump(
                    output,
                    f,
                    indent=2
                )


            print()
            print(
                "saved:",
                OUT
            )

            print()
            print(
                "LAST FRAME DISTANCES"
            )

            print(
                f"agent11-agent12 = "
                f"{d12:.2f} m"
            )

            print(
                f"agent11-agent13 = "
                f"{d13:.2f} m"
            )

            print(
                f"agent12-agent13 = "
                f"{d23:.2f} m"
            )

            print()
            print(
                "Recording complete."
            )

            rclpy.shutdown()


def main():

    rclpy.init()

    node = JointWindowRecorder()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
