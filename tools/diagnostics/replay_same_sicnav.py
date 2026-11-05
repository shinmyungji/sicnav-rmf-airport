#!/usr/bin/env python3

import json
import socket
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


SOCKET_PATH = "/tmp/sicnav_acados.sock"

GOAL = {
    "x": 141.0,
    "y": -41.7,
}


class ReplayTest(Node):

    def __init__(self):
        super().__init__("sicnav_same_snapshot_test")

        self.robot = None
        self.humans = None
        self.prediction = None
        self.captured = False

        self.create_subscription(
            String,
            "/sicnav/robot_state",
            self.robot_cb,
            10
        )

        self.create_subscription(
            String,
            "/sicnav/human_states",
            self.human_cb,
            10
        )

        self.create_subscription(
            String,
            "/sicnav/jmid_predictions",
            self.pred_cb,
            10
        )

        print("Waiting for ONE fixed snapshot...")


    def robot_cb(self, msg):
        if not self.captured:
            self.robot = json.loads(msg.data)
            self.try_capture()


    def human_cb(self, msg):
        if not self.captured:
            self.humans = json.loads(msg.data)
            self.try_capture()


    def pred_cb(self, msg):
        if not self.captured:
            self.prediction = json.loads(msg.data)
            self.try_capture()


    def try_capture(self):

        if self.captured:
            return

        if self.robot is None:
            return

        if self.humans is None:
            return

        if self.prediction is None:
            return

        if "forecast" not in self.prediction:
            return

        if "log_prob" not in self.prediction:
            return

        if "rmf_ids" not in self.prediction:
            self.prediction["rmf_ids"] = [11, 12, 13]

        self.captured = True

        request = {
            "cmd": "solve",
            "robot": self.robot,
            "humans": self.humans,
            "prediction": self.prediction,
            "goal": GOAL,
        }

        print()
        print("======================================")
        print("FIXED SNAPSHOT CAPTURED")
        print("======================================")
        print("robot:", self.robot)
        print("JMID request:", self.prediction.get("request_id"))
        print()

        for i in range(1, 6):

            sock = socket.socket(
                socket.AF_UNIX,
                socket.SOCK_STREAM
            )

            sock.settimeout(30.0)
            sock.connect(SOCKET_PATH)

            payload = (
                json.dumps(
                    request,
                    separators=(",", ":")
                )
                + "\n"
            ).encode()

            start = time.perf_counter()

            sock.sendall(payload)

            data = b""

            while b"\n" not in data:

                chunk = sock.recv(65536)

                if not chunk:
                    break

                data += chunk

            wall_ms = (
                time.perf_counter() - start
            ) * 1000.0

            sock.close()

            result = json.loads(
                data.split(b"\n", 1)[0].decode()
            )

            print(
                f"SOLVE {i}: "
                f"v={result.get('v')}  "
                f"w={result.get('omega')}  "
                f"solve={result.get('solve_ms')} ms  "
                f"wall={wall_ms:.1f} ms  "
                f"status={result.get('acados_status')}  "
                f"iter={result.get('sqp_iter')}  "
                f"success={result.get('mpc_success')}"
            )

        print()
        print("DONE")

        rclpy.shutdown()


def main():

    rclpy.init()

    node = ReplayTest()

    try:
        rclpy.spin(node)

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
