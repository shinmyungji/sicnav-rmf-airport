#!/usr/bin/env python3

import os
import sys
import json
import socket
import configparser
import traceback

import numpy as np

REPO = "/home/qwer1234/sicnav/upstream/safe-interactive-crowdnav"
SOCKET_PATH = "/tmp/sicnav_jmid.sock"

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


def recv_line(conn):
    data = b""

    while True:
        chunk = conn.recv(65536)

        if not chunk:
            break

        data += chunk

        if b"\n" in data:
            break

    return data.split(b"\n", 1)[0]


def send_json(conn, obj):
    payload = (
        json.dumps(obj, separators=(",", ":"))
        + "\n"
    ).encode()

    conn.sendall(payload)


# ==========================================================
# JMID MODEL LOAD — 딱 한 번
# ==========================================================

print("==============================")
print("Persistent JMID Worker")
print("==============================")
print("Loading JMID model ONCE...")

env_config = configparser.RawConfigParser()
env_config.read(ENV_CONFIG)

# 지금 실험은 3 humans 고정
env_config.set(
    "sim",
    "human_num",
    "3"
)

forecaster = HumanTrajectoryForecasterSim(
    env_config=env_config,
    mid_config_file=MID_CONFIG
)

print("JMID READY")
print()


# ==========================================================
# UNIX socket
# ==========================================================

if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

server = socket.socket(
    socket.AF_UNIX,
    socket.SOCK_STREAM
)

server.bind(SOCKET_PATH)
server.listen(5)

print("Listening:")
print(SOCKET_PATH)
print()

request_count = 0


try:

    while True:

        conn, _ = server.accept()

        try:

            raw = recv_line(conn)

            if not raw:
                conn.close()
                continue

            req = json.loads(
                raw.decode()
            )

            if req.get("cmd") == "ping":

                send_json(
                    conn,
                    {
                        "ok": True,
                        "status": "JMID_READY"
                    }
                )

                continue


            if req.get("cmd") != "predict":

                send_json(
                    conn,
                    {
                        "ok": False,
                        "error": "unknown command"
                    }
                )

                continue


            history = req["history"]

            frames = history["frames"]

            num_humans = int(
                history["num_humans"]
            )

            if num_humans != 3:

                raise RuntimeError(
                    f"worker currently expects 3 humans, "
                    f"got {num_humans}"
                )


            # ==============================================
            # 기존 요청의 history 완전히 제거
            # ==============================================

            forecaster.prev_states = [
                []
                for _ in range(num_humans)
            ]

            forecaster.prev_robot_states = []


            # ==============================================
            # JSON history → JMID history
            # ==============================================

            for frame in frames:

                t = float(
                    frame["time"]
                )

                humans = sorted(
                    frame["humans"],
                    key=lambda h: h["jmid_id"]
                )

                for h in humans:

                    idx = int(
                        h["jmid_id"]
                    )

                    forecaster.prev_states[
                        idx
                    ].append(
                        [
                            float(h["x"]),
                            float(h["y"]),
                            t
                        ]
                    )


                robot = frame["robot"]

                forecaster.prev_robot_states.append(
                    [
                        float(robot["x"]),
                        float(robot["y"]),
                        t
                    ]
                )


            # ==============================================
            # JMID inference
            # ==============================================

            request_count += 1

            print(
                f"[request {request_count}] "
                f"running inference..."
            )

            forecast, log_prob = (
                forecaster.predict_ret_best()
            )

            forecast = np.asarray(
                forecast
            )

            log_prob = np.asarray(
                log_prob
            )

            print(
                f"[request {request_count}] "
                f"forecast={forecast.shape}, "
                f"log_prob={log_prob.shape}"
            )


            send_json(
                conn,
                {
                    "ok": True,

                    "request_id":
                        request_count,

                    "forecast_shape":
                        list(forecast.shape),

                    "log_prob_shape":
                        list(log_prob.shape),

                    "forecast":
                        forecast.tolist(),

                    "log_prob":
                        log_prob.tolist()
                }
            )


        except Exception as e:

            traceback.print_exc()

            try:
                send_json(
                    conn,
                    {
                        "ok": False,
                        "error": str(e)
                    }
                )
            except Exception:
                pass


        finally:
            conn.close()


finally:

    server.close()

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
