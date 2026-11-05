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

N_REPLAY = 5


class Capture(Node):

    def __init__(self):
        super().__init__("sicnav_fixed_request_capture")

        self.prediction = None

        self.create_subscription(
            String,
            "/sicnav/jmid_predictions",
            self.cb,
            10,
        )

        self.get_logger().info(
            "Waiting for ONE synchronized JMID packet..."
        )

    def cb(self, msg):

        if self.prediction is not None:
            return

        pred = json.loads(msg.data)

        if not pred.get("ok", False):
            return

        if "humans_t0" not in pred or "robot_t0" not in pred:
            return

        self.prediction = pred

        print()
        print("====================================")
        print("FIXED REQUEST CAPTURED")
        print("====================================")
        print("request_id :", pred.get("request_id"))
        print("sim_time_t0:", pred.get("sim_time_t0"))
        print("robot_t0   :", pred["robot_t0"])
        print("humans_t0  :", pred["humans_t0"])
        print()

        rclpy.shutdown()


def send_request(request):

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

    return json.loads(
        data.split(b"\n", 1)[0].decode()
    )


def main():

    rclpy.init()

    node = Capture()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    pred = node.prediction

    node.destroy_node()

    if pred is None:
        raise RuntimeError(
            "No JMID prediction captured"
        )

    request = {
        "cmd": "solve",
        "robot": pred["robot_t0"],
        "humans": pred["humans_t0"],
        "prediction": pred,
        "goal": GOAL,
    }

    # Save exact request for later inspection
    save_path = "/tmp/sicnav_fixed_request.json"

    with open(save_path, "w") as f:
        json.dump(
            request,
            f,
            indent=2
        )

    print(
        "Exact request saved:",
        save_path
    )

    print()
    print("====================================")
    print("SAME REQUEST REPLAY")
    print("====================================")

    for i in range(1, N_REPLAY + 1):

        result = send_request(request)

        print()
        print(f"---------- REPLAY {i} ----------")
        print(json.dumps(
            result,
            indent=2
        ))


if __name__ == "__main__":
    main()
