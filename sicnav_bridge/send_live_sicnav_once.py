#!/usr/bin/env python3

import json
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


SOCKET_PATH = "/tmp/sicnav_acados.sock"

GOAL = {
    "x": 141.0,
    "y": -41.7,
}


class SICNavOneShot(Node):

    def __init__(self):
        super().__init__("sicnav_one_shot_client")

        self.prediction = None
        self.sent = False

        # IMPORTANT:
        # Do NOT subscribe to latest robot/human states here.
        #
        # robot_t0 / humans_t0 must be the exact snapshot that
        # generated the JMID prediction.
        self.create_subscription(
            String,
            "/sicnav/jmid_predictions",
            self.pred_cb,
            10,
        )

        self.get_logger().info(
            "Waiting for synchronized JMID prediction packet..."
        )


    def pred_cb(self, msg):

        if self.sent:
            return

        try:
            self.prediction = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(
                f"prediction JSON error: {e}"
            )
            return

        self.try_send()


    def try_send(self):

        if self.sent:
            return

        if self.prediction is None:
            return

        required = [
            "forecast",
            "log_prob",
            "rmf_ids",
            "robot_t0",
            "humans_t0",
        ]

        missing = [
            key
            for key in required
            if key not in self.prediction
        ]

        if missing:
            self.get_logger().error(
                f"JMID packet missing fields: {missing}"
            )
            return

        robot_t0 = self.prediction["robot_t0"]
        humans_t0 = self.prediction["humans_t0"]

        if len(humans_t0) != 3:
            self.get_logger().error(
                f"Expected 3 synchronized humans, got {len(humans_t0)}"
            )
            return

        self.sent = True

        request = {
            "cmd": "solve",

            # SAME state that generated this JMID prediction
            "robot": robot_t0,
            "humans": humans_t0,

            "prediction": self.prediction,
            "goal": GOAL,
        }

        print()
        print("====================================")
        print("SYNCED AIRPORT -> SICNAV ONE SHOT")
        print("====================================")
        print("robot_t0:", robot_t0)
        print("humans_t0:", humans_t0)
        print("goal:", GOAL)
        print(
            "JMID request_id:",
            self.prediction.get("request_id", "unknown")
        )
        print(
            "capture_wall_time:",
            self.prediction.get("capture_wall_time", "unknown")
        )
        print(
            "rmf_ids:",
            self.prediction["rmf_ids"]
        )
        print()

        try:
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

            sock.sendall(payload)

            data = b""

            while b"\n" not in data:
                chunk = sock.recv(65536)

                if not chunk:
                    break

                data += chunk

            sock.close()

            if not data:
                raise RuntimeError(
                    "empty response from SICNav server"
                )

            result = json.loads(
                data.split(b"\n", 1)[0].decode()
            )

            print()
            print("====================================")
            print("SICNAV RESULT")
            print("====================================")
            print(
                json.dumps(
                    result,
                    indent=2
                )
            )
            print()

        except Exception as e:
            print()
            print(
                "SICNav request FAILED:",
                repr(e)
            )
            print()

        rclpy.shutdown()


def main():

    rclpy.init()

    node = SICNavOneShot()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
