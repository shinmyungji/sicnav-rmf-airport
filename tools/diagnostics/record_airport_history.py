#!/usr/bin/env python3

import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


NUM_FRAMES = 6
DT = 0.25
OUT = "/home/qwer1234/sicnav/logs/airport_history.json"


class HistoryRecorder(Node):

    def __init__(self):
        super().__init__("airport_history_recorder")

        self.latest_humans = None
        self.human_seq = 0
        self.last_saved_human_seq = -1

        self.original_ids = None
        self.frames = []

        self.create_subscription(
            String,
            "/sicnav/human_states",
            self.human_cb,
            10,
        )

        self.create_subscription(
            String,
            "/sicnav/robot_state",
            self.robot_cb,
            10,
        )

        self.get_logger().info(
            f"Recording {NUM_FRAMES} frames at dt={DT}s"
        )

    def human_cb(self, msg):
        self.latest_humans = json.loads(msg.data)
        self.human_seq += 1

    def robot_cb(self, msg):

        if self.latest_humans is None:
            return

        # 같은 human frame을 두 번 저장하지 않음
        if self.human_seq == self.last_saved_human_seq:
            return

        humans = sorted(
            self.latest_humans,
            key=lambda h: h["id"]
        )

        ids = [h["id"] for h in humans]

        # 첫 frame에서 RMF ID 순서 고정
        if self.original_ids is None:
            self.original_ids = ids

            print("\nRMF -> JMID ID mapping")
            for jmid_id, rmf_id in enumerate(self.original_ids):
                print(f"  agent{rmf_id} -> {jmid_id}")

        # 사람이 사라지거나 새로 생겼으면 history를 다시 쌓음
        if ids != self.original_ids:
            self.get_logger().warning(
                "Human ID set changed. Resetting history."
            )

            self.original_ids = ids
            self.frames = []

            return

        robot = json.loads(msg.data)

        frame_idx = len(self.frames)

        frame = {
            "time": frame_idx * DT,

            "humans": [
                {
                    # JMID 내부에서는 0..N-1 필요
                    "jmid_id": idx,

                    # 나중에 결과를 agent11 등에 다시 대응
                    "rmf_id": h["id"],

                    "x": h["x"],
                    "y": h["y"],
                }
                for idx, h in enumerate(humans)
            ],

            "robot": {
                "x": robot["x"],
                "y": robot["y"],
                "yaw": robot["yaw"],
            }
        }

        self.frames.append(frame)
        self.last_saved_human_seq = self.human_seq

        print(
            f"frame {len(self.frames)}/{NUM_FRAMES}  "
            f"humans={len(humans)}  "
            f"robot=({robot['x']:.2f},{robot['y']:.2f})"
        )

        if len(self.frames) >= NUM_FRAMES:
            self.save()

    def save(self):

        output = {
            "dt": DT,
            "num_frames": NUM_FRAMES,
            "num_humans": len(self.original_ids),

            "rmf_ids": self.original_ids,

            "id_mapping": {
                str(rmf_id): jmid_id
                for jmid_id, rmf_id
                in enumerate(self.original_ids)
            },

            "frames": self.frames,
        }

        with open(OUT, "w") as f:
            json.dump(output, f, indent=2)

        print("\n==============================")
        print("History recording complete")
        print("==============================")
        print(f"saved: {OUT}")
        print(f"humans: {len(self.original_ids)}")
        print(f"frames: {len(self.frames)}")

        rclpy.shutdown()


def main():

    rclpy.init()

    node = HistoryRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
