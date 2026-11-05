#!/usr/bin/env python3

import json
import socket
import time
import threading
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


SOCKET_PATH = "/tmp/sicnav_jmid.sock"

DT = 0.25
NUM_FRAMES = 6
EXPECTED_HUMANS = 3


class LiveJMIDBridge(Node):

    def __init__(self):
        super().__init__("live_jmid_bridge")

        # --------------------------------------------------
        # Latest Gazebo/RMF states
        # --------------------------------------------------
        self.latest_humans = None
        self.latest_robot = None
        self.latest_sim_time = None

        self.rmf_ids = None
        self.id_mapping = None

        # rolling 6-frame history
        self.buffer = deque(maxlen=NUM_FRAMES)

        self.state_lock = threading.Lock()

        # --------------------------------------------------
        # Async JMID inference
        # --------------------------------------------------
        self.infer_pool = ThreadPoolExecutor(max_workers=1)
        self.inference_running = False

        self.result_queue = queue.Queue()

        # --------------------------------------------------
        # ROS subscribers
        # --------------------------------------------------
        self.create_subscription(
            String,
            "/sicnav/state_frame",
            self.frame_cb,
            10
        )

        # --------------------------------------------------
        # ROS publishers
        # --------------------------------------------------
        self.pred_pub = self.create_publisher(
            String,
            "/sicnav/jmid_predictions",
            10
        )

        self.status_pub = self.create_publisher(
            String,
            "/sicnav/jmid_status",
            10
        )

        # publish finished async results
        self.create_timer(
            0.05,
            self.result_timer_cb
        )

        print()
        print("==============================")
        print("LIVE JMID BRIDGE")
        print("==============================")
        print(f"history frames : {NUM_FRAMES}")
        print(f"dt             : {DT} sec")
        print(f"socket         : {SOCKET_PATH}")
        print()
        print("Waiting for human + robot states...")
        print()


    # ======================================================
    # ROS state callbacks
    # ======================================================

    def frame_cb(self, msg):

        try:
            frame = json.loads(msg.data)

            humans = frame["humans"]
            robot = frame["robot"]
            sim_time = float(frame["sim_time"])

            with self.state_lock:
                self.latest_humans = humans
                self.latest_robot = robot
                self.latest_sim_time = sim_time

        except Exception as e:
            self.get_logger().error(
                f"state_frame parse error: {e}"
            )
            return

        # One Gazebo simulation-time frame -> one history update.
        # No independent wall-clock resampling here.
        self.sample_timer_cb()


    # ======================================================
    # 0.25 sec sampling
    # ======================================================

    def sample_timer_cb(self):

        with self.state_lock:

            if (
                self.latest_humans is None
                or self.latest_robot is None
                or self.latest_sim_time is None
            ):
                return

            humans = sorted(
                self.latest_humans,
                key=lambda h: h["id"]
            )

            robot = dict(
                self.latest_robot
            )

            sim_time = float(
                self.latest_sim_time
            )


        if len(humans) != EXPECTED_HUMANS:

            print(
                f"\rWaiting: humans={len(humans)}",
                end="",
                flush=True
            )

            return


        # --------------------------------------------------
        # Stable ID mapping
        # --------------------------------------------------

        current_ids = [
            h["id"]
            for h in humans
        ]


        if self.rmf_ids is None:

            self.rmf_ids = current_ids

            self.id_mapping = {
                rid: idx
                for idx, rid
                in enumerate(self.rmf_ids)
            }

            print("RMF -> JMID mapping")

            for rid, idx in self.id_mapping.items():
                print(
                    f"  agent{rid} -> {idx}"
                )

            print()


        elif current_ids != self.rmf_ids:

            print()
            print(
                "Human IDs changed. "
                "Resetting history."
            )

            self.rmf_ids = current_ids

            self.id_mapping = {
                rid: idx
                for idx, rid
                in enumerate(self.rmf_ids)
            }

            self.buffer.clear()


        # --------------------------------------------------
        # Store current frame
        # --------------------------------------------------

        frame = {
            "sim_time": float(sim_time),

            "humans": [
                {
                    "jmid_id":
                        self.id_mapping[h["id"]],

                    "rmf_id":
                        h["id"],

                    "x":
                        float(h["x"]),

                    "y":
                        float(h["y"])
                }

                for h in humans
            ],

            "robot": {
                "x":
                    float(robot["x"]),

                "y":
                    float(robot["y"]),

                "yaw":
                    float(
                        robot.get(
                            "yaw",
                            0.0
                        )
                    )
            }
        }


        self.buffer.append(frame)


        print(
            "\r"
            f"history={len(self.buffer)}/{NUM_FRAMES}  "
            f"inference={'BUSY' if self.inference_running else 'READY'}",
            end="",
            flush=True
        )


        # Need full 6-frame history
        if len(self.buffer) < NUM_FRAMES:
            return


        # Do not build inference backlog
        if self.inference_running:
            return


        # --------------------------------------------------
        # Make fresh 6-frame request
        # --------------------------------------------------

        frames = []

        for i, stored in enumerate(self.buffer):

            frames.append(
                {
                    "time":
                        float(stored["sim_time"]),

                    "humans":
                        stored["humans"],

                    "robot":
                        stored["robot"]
                }
            )


        history = {
            "dt":
                DT,

            "num_frames":
                NUM_FRAMES,

            "num_humans":
                EXPECTED_HUMANS,

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


        # --------------------------------------------------
        # EXACT state snapshot used for this JMID request
        #
        # This MUST travel together with the corresponding
        # JMID prediction. SICNav must NOT combine the prediction
        # with a newer ROS state.
        # --------------------------------------------------
        snapshot_t0 = {
            "humans": [
                {
                    "id": int(h["id"]),
                    "x": float(h["x"]),
                    "y": float(h["y"]),
                    "vx": float(h.get("vx", 0.0)),
                    "vy": float(h.get("vy", 0.0)),
                }
                for h in humans
            ],

            "robot": {
                "x": float(robot["x"]),
                "y": float(robot["y"]),
                "yaw": float(robot.get("yaw", 0.0)),
                "v": float(robot.get("v", 0.0)),
                "omega": float(robot.get("omega", 0.0)),
            },

            # Diagnostic timestamp only.
            # The important synchronization is that state and
            # prediction originate from this same captured frame.
            "capture_wall_time": time.time(),
            "sim_time": float(sim_time),
        }

        self.inference_running = True

        self.infer_pool.submit(
            self.run_jmid_request,
            history,
            snapshot_t0
        )


    # ======================================================
    # Socket -> persistent Python3.8 JMID worker
    # ======================================================

    def run_jmid_request(self, history, snapshot_t0):

        start = time.perf_counter()

        try:

            req = {
                "cmd": "predict",
                "history": history
            }


            sock = socket.socket(
                socket.AF_UNIX,
                socket.SOCK_STREAM
            )

            sock.settimeout(30.0)

            sock.connect(
                SOCKET_PATH
            )


            payload = (
                json.dumps(
                    req,
                    separators=(",", ":")
                )
                + "\n"
            ).encode()


            sock.sendall(payload)


            data = b""

            while True:

                chunk = sock.recv(
                    65536
                )

                if not chunk:
                    break

                data += chunk

                if b"\n" in data:
                    break


            sock.close()


            response = json.loads(
                data.split(
                    b"\n",
                    1
                )[0]
            )


            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000.0


            if not response.get("ok", False):

                raise RuntimeError(
                    response.get(
                        "error",
                        "JMID worker error"
                    )
                )


            result = {
                "ok":
                    True,

                "request_id":
                    response["request_id"],

                "rmf_ids":
                    list(history["rmf_ids"]),

                # Exact Gazebo/RMF state corresponding to
                # the final history frame used by this JMID request.
                "humans_t0":
                    snapshot_t0["humans"],

                "robot_t0":
                    snapshot_t0["robot"],

                "capture_wall_time":
                    snapshot_t0["capture_wall_time"],

                "sim_time_t0":
                    snapshot_t0["sim_time"],

                "forecast_shape":
                    response["forecast_shape"],

                "log_prob_shape":
                    response["log_prob_shape"],

                "forecast":
                    response["forecast"],

                "log_prob":
                    response["log_prob"],

                "latency_ms":
                    latency_ms
            }


            self.result_queue.put(
                result
            )


        except Exception as e:

            self.result_queue.put(
                {
                    "ok": False,
                    "error": str(e)
                }
            )


        finally:

            self.inference_running = False


    # ======================================================
    # ROS publication
    # ======================================================

    def result_timer_cb(self):

        while not self.result_queue.empty():

            result = self.result_queue.get()


            if not result.get("ok", False):

                self.get_logger().error(
                    "JMID request failed: "
                    + result.get(
                        "error",
                        "unknown error"
                    )
                )

                continue


            # ------------------------------------------------
            # Full prediction
            # ------------------------------------------------

            msg = String()

            msg.data = json.dumps(
                result,
                separators=(",", ":")
            )

            self.pred_pub.publish(
                msg
            )


            # ------------------------------------------------
            # Small debug/status topic
            # ------------------------------------------------

            status = {
                "request_id":
                    result["request_id"],

                "rmf_ids":
                    result["rmf_ids"],

                "forecast_shape":
                    result["forecast_shape"],

                "log_prob_shape":
                    result["log_prob_shape"],

                "latency_ms":
                    round(
                        result["latency_ms"],
                        2
                    )
            }


            status_msg = String()

            status_msg.data = json.dumps(
                status
            )

            self.status_pub.publish(
                status_msg
            )


            print()
            print(
                f"JMID request "
                f"{result['request_id']} OK | "
                f"{result['forecast_shape']} | "
                f"{result['latency_ms']:.1f} ms"
            )


def main():

    rclpy.init()

    node = LiveJMIDBridge()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.infer_pool.shutdown(
            wait=False
        )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
