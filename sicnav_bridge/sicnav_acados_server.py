#!/usr/bin/env python3

import os
import sys
import runpy
import json
import socket
import time
import math
import traceback

import numpy as np

REPO = "/home/qwer1234/sicnav/upstream/safe-interactive-crowdnav"
INIT_SCRIPT = "/home/qwer1234/sicnav/scripts/sicnav_acados_worker.py"
SOCKET_PATH = "/tmp/sicnav_acados.sock"

os.chdir(REPO)
sys.path.insert(0, REPO)

from crowd_sim_plus.envs.utils.state_plus import (
    FullState,
    FullyObservableJointState,
)


print()
print("====================================")
print("Persistent SICNav / Acados Server")
print("====================================")
print("Loading cached SICNav solver...")
print()


# 기존에 성공한 초기화 스크립트 실행
ctx = runpy.run_path(INIT_SCRIPT)

policy = ctx["policy"]
env_config = ctx["env_config"]
policy_config = ctx["policy_config"]


# dummy solve 결과는 실제 Airport에 사용하지 않음
policy.reset_scenario_values()




def normalize_humans(obj):
    """
    /sicnav/human_states JSON 형식이
    list 또는 {"humans": [...]} 둘 다 대응.
    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        if "humans" in obj:
            return obj["humans"]

        if "states" in obj:
            return obj["states"]

    raise ValueError(
        "Unknown human_states JSON format"
    )


def solve_request(req):

    robot = req["robot"]
    humans_raw = normalize_humans(
        req["humans"]
    )
    pred = req["prediction"]
    goal = req["goal"]

    forecast = np.asarray(
        pred["forecast"],
        dtype=np.float64
    )

    log_prob = np.asarray(
        pred["log_prob"],
        dtype=np.float64
    )

    rmf_ids = [
        int(x)
        for x in pred["rmf_ids"]
    ]


    # --------------------------------------------------
    # Shape validation
    # --------------------------------------------------

    if forecast.shape != (3, 15, 9, 2):
        raise ValueError(
            "forecast shape is "
            + str(forecast.shape)
        )

    if log_prob.shape != (3, 15):
        raise ValueError(
            "log_prob shape is "
            + str(log_prob.shape)
        )


    # --------------------------------------------------
    # 사람 순서를 JMID index와 정확히 맞춤
    #
    # rmf_ids = [11,12,13]
    # forecast[0] = agent11
    # forecast[1] = agent12
    # forecast[2] = agent13
    # --------------------------------------------------

    human_by_id = {
        int(h["id"]): h
        for h in humans_raw
    }

    humans = [
        human_by_id[h_id]
        for h_id in rmf_ids
    ]


    # --------------------------------------------------
    # Original SICNav:
    #
    # forecasts = top_k_forecasts[:,:,1:,:]
    #
    # t0 제거:
    # [3,15,9,2] -> [3,15,8,2]
    # --------------------------------------------------

    forecasts = forecast[:, :, 1:, :]


    # joint JMID는 15개의 joint log-weight 사용
    forecasts_init_weights = log_prob[0, :]


    # --------------------------------------------------
    # Robot FullState
    # --------------------------------------------------

    yaw = float(robot["yaw"])
    v = float(robot.get("v", 0.0))
    omega = float(robot.get("omega", 0.0))

    robot_state = FullState(
        px=float(robot["x"]),
        py=float(robot["y"]),

        vx=v * math.cos(yaw),
        vy=v * math.sin(yaw),

        radius=env_config.getfloat(
            "robot",
            "radius"
        ),

        gx=float(goal["x"]),
        gy=float(goal["y"]),

        v_pref=policy.pref_vel_rob,

        theta=yaw,
        omega=omega,
    )

    # 원본 convert_to_mpc_state_vector가 사용
    robot_state.lvel = v
    robot_state.v_dot = 0.0
    robot_state.omega_dot = 0.0


    # --------------------------------------------------
    # Human FullState
    #
    # 원본 SICNav predict()와 동일:
    # - JMID 첫 미래점 평균 → pseudo goal
    # - prediction 속도 → v_pref
    # --------------------------------------------------

    human_states = []

    human_radius = policy.human_radius

    for h_idx, h in enumerate(humans):

        gx = float(
            np.mean(
                forecasts[
                    h_idx, :, 0, 0
                ]
            )
        )

        gy = float(
            np.mean(
                forecasts[
                    h_idx, :, 0, 1
                ]
            )
        )

        diffs = np.diff(
            forecasts[
                h_idx, :, :, :
            ],
            axis=1
        )

        speeds = (
            np.linalg.norm(
                diffs,
                axis=2
            )
            /
            policy.time_step
        )

        v_pref = float(
            np.max(speeds)
        )

        vx = float(h.get("vx", 0.0))
        vy = float(h.get("vy", 0.0))

        theta = (
            math.atan2(vy, vx)
            if abs(vx) + abs(vy) > 1e-8
            else 0.0
        )

        human_states.append(
            FullState(
                px=float(h["x"]),
                py=float(h["y"]),

                vx=vx,
                vy=vy,

                radius=human_radius,

                gx=gx,
                gy=gy,

                v_pref=v_pref,

                theta=theta,
            )
        )


    # 첫 실험에서는 Airport 벽 제외.
    # 사람 interaction만 실제 데이터로 검증.
    joint_state = FullyObservableJointState(
        robot_state,
        human_states,
        []
    )


    # --------------------------------------------------
    # Robot reference trajectory
    # --------------------------------------------------

    policy.gen_ref_traj(
        joint_state
    )

    goal_states, goal_actions = (
        policy.get_ref_traj(
            joint_state
        )
    )


    # --------------------------------------------------
    # SICNav state vector
    # --------------------------------------------------

    mpc_state = (
        policy.convert_to_mpc_state_vector(
            joint_state,

            policy.mpc_env.nx_r,
            policy.mpc_env.np_g,
            policy.mpc_env.nX_hums,

            forecasts_init_weights,

            get_numpy=True,
        )
    )


    # --------------------------------------------------
    # JMID:
    #
    # [3,15,8,2]
    #       ↓
    # t,(human*sample),xy
    #       ↓
    # [8,45,2]
    #
    # horiz=4 → t0~t4 필요
    #       ↓
    # [5,45,2]
    # --------------------------------------------------

    MID_samples = (
        forecasts
        .transpose(2, 0, 1, 3)
        .reshape(
            forecasts.shape[2],
            forecasts.shape[0]
            * forecasts.shape[1],
            2
        )
    )

    MID_samples = MID_samples[
        :policy.horiz + 1
    ]


    # --------------------------------------------------
    # 실제 SICNav / Acados solve
    # --------------------------------------------------

    start = time.perf_counter()

    mpc_action = policy.select_action(
        mpc_state,
        joint_state,
        goal_states,
        goal_actions,

        MID_samples=MID_samples,
    )

    solve_ms = (
        time.perf_counter() - start
    ) * 1000.0


    v_cmd = float(
        mpc_action[0]
    )

    omega_cmd = float(
        mpc_action[1]
    )


    # original predict()에서 하는 update
    policy.prev_lvel = v_cmd
    policy.traj_step += 1


    result = {
        "ok": True,

        "v": v_cmd,
        "omega": omega_cmd,

        "solve_ms": round(
            solve_ms,
            3
        ),

        "acados_status": int(
            policy.prev_status
        ),

        "sqp_iter": int(
            policy.prev_num_iter
        ),

        "mpc_success": bool(
            policy.mpc_sol_succ
        ),

        "mid_shape": list(
            MID_samples.shape
        ),

        "rmf_ids": rmf_ids,
    }

    print()
    print("====================================")
    print("LIVE SICNAV SOLVE")
    print("====================================")
    print("v            :", v_cmd)
    print("omega        :", omega_cmd)
    print("solve_ms     :", solve_ms)
    print("status       :", policy.prev_status)
    print("SQP iter     :", policy.prev_num_iter)
    print("MPC success  :", policy.mpc_sol_succ)
    print("MID shape    :", MID_samples.shape)
    print()

    return result


# ============================================================
# UNIX socket server
# ============================================================

if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

server = socket.socket(
    socket.AF_UNIX,
    socket.SOCK_STREAM
)

server.bind(SOCKET_PATH)
server.listen(4)


print()
print("====================================")
print("SICNAV SERVER READY")
print("====================================")
print("socket :", SOCKET_PATH)
print("TB4 command publishing: DISABLED")
print()


def recv_line(conn):

    data = b""

    while True:

        chunk = conn.recv(65536)

        if not chunk:
            break

        data += chunk

        if b"\n" in data:
            break

    return data.split(
        b"\n",
        1
    )[0]


def send_json(conn, obj):

    conn.sendall(
        (
            json.dumps(
                obj,
                separators=(",", ":")
            )
            + "\n"
        ).encode()
    )


try:

    while True:

        conn, _ = server.accept()

        try:

            raw = recv_line(conn)

            if not raw:
                continue

            req = json.loads(
                raw.decode()
            )

            if req.get("cmd") == "ping":

                send_json(
                    conn,
                    {
                        "ok": True,
                        "status":
                            "SICNAV_READY"
                    }
                )

            elif req.get("cmd") == "solve":

                send_json(
                    conn,
                    solve_request(req)
                )

            else:

                send_json(
                    conn,
                    {
                        "ok": False,
                        "error":
                            "unknown command"
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
