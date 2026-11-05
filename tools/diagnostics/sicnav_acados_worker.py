#!/usr/bin/env python3

import os
import sys
import configparser

import torch
import gym

REPO = "/home/qwer1234/sicnav/upstream/safe-interactive-crowdnav"

os.chdir(REPO)
sys.path.insert(0, REPO)

from crowd_sim_plus.envs.crowd_sim_plus import *
from crowd_sim_plus.envs.utils.robot_plus import Robot
from sicnav_diffusion.policy.sicnav_acados import SICNavAcados


ENV_CONFIG = os.path.join(
    REPO,
    "sicnav_diffusion/configs/env.config"
)

POLICY_CONFIG = os.path.join(
    REPO,
    "sicnav_diffusion/configs/policy.config"
)


# ============================================================
# External JMID용 SICNav
#
# 원본 reset_scenario_values()는 내부에서 JMID를 다시 load한다.
# 우리는 이미 별도 persistent JMID worker를 사용하므로
# 여기서는 그 부분만 제거한다.
# ============================================================

class ExternalJMIDSICNav(SICNavAcados):

    def reset_scenario_values(self):

        self.traj_step = 0

        self.prev_lvel = 0.0
        self.prev_avel = 0.0

        self.x_prev = None
        self.u_prev = None

        self.num_prev_used = 0
        self.mpc_progress = None

        self.all_goal_diff = []

        self.all_x_val = []
        self.all_u_val = []

        self.all_x_guess = []
        self.all_u_guess = []

        self.all_x_goals = []
        self.all_u_goals = []

        self.all_debug_text = []

        self.mpc_sol_succ = []

        self.calc_times = []

        self.all_prev_status = []
        self.all_prev_num_iter = []

        self.solver_summary = {
            "traj_step": [],
            "sol_success": [],
            "optim_status": [],
            "iter_count": [],
            "prep_time": [],
            "sol_time": [],
            "final_nopenal_cost": [],
            "final_term_cost": [],
            "debug_text": [],
            "ipopt_iterations": [],
        }

        # 원본에서는 여기서 init_human_prediction() 호출함.
        # 우리는 외부 JMID를 쓰므로 호출하지 않음.
        if self.human_pred_MID:

            self.all_forecasts = []
            self.all_weights = []
            self.all_cum_weights = []

            self.all_forecasts_shaped = []

            self.all_log_weights = []
            self.all_log_weights_unnormed = []


print()
print("====================================")
print("SICNav / Acados initialization test")
print("====================================")


# ============================================================
# config
# ============================================================

env_config = configparser.RawConfigParser()
env_config.read(ENV_CONFIG)

# 현재 RMF crowd와 동일
env_config.set(
    "sim",
    "human_num",
    "3"
)

# solver 초기화용 CrowdSim scenario
env_config.set(
    "sim",
    "test_sim",
    "circle_crossing"
)

# 첫 테스트에서는 Airport static obstacle를 아직 안 넣음
env_config.set(
    "env",
    "wild",
    "false"
)

env_config.set(
    "robot",
    "policy",
    "none"
)


policy_config = configparser.RawConfigParser()
policy_config.read(POLICY_CONFIG)


print("human_num :", env_config.getint("sim", "human_num"))
print("time_step :", env_config.getfloat("env", "time_step"))

print(
    "human_pred_MID :",
    policy_config.getboolean(
        "campc",
        "human_pred_MID"
    )
)

print(
    "JMID joint     :",
    policy_config.getboolean(
        "campc",
        "human_pred_MID_joint"
    )
)

print(
    "human model    :",
    policy_config.get(
        "mpc_env",
        "hum_model"
    )
)

print(
    "MPC horizon    :",
    policy_config.getint(
        "campc",
        "horiz"
    )
)

print()


# ============================================================
# dummy CrowdSim env
#
# 실제 제어에 CrowdSim을 쓰는 게 아님.
# 원본 SICNav set_env()가 solver 구조를 만들 때
# 필요한 초기 state를 공급하기 위한 용도.
# ============================================================

print("[1/4] Creating initialization environment...")

env_order = gym.make(
    "CrowdSimPlus-v0"
)

env = env_order.unwrapped

env.configure(
    env_config
)

# CrowdSimPlus square_crossing compatibility:
# configure() stores this in env.rewards, while the old
# square-crossing generator still reads env.discomfort_dist.
env.discomfort_dist = env.rewards.get("discomfort_dist", 0.2)

env.test_sim = "circle_crossing"


# ============================================================
# policy
# ============================================================

print("[2/4] Creating SICNav policy...")

policy = ExternalJMIDSICNav()

policy.configure(
    policy_config
)


robot = Robot(
    env_config,
    "robot"
)

robot.set_policy(
    policy
)

env.set_robot(
    robot
)

policy.set_phase(
    "test"
)

policy.set_device(
    torch.device("cpu")
)


# ============================================================
# THIS builds:
#
# MPCEnv
# ORCA CasADi model
# KKT constraints
# Acados OCP
# generated C code / cached solver
# warmstart
# ============================================================

print()
print("[3/4] Building ORCA/KKT/Acados...")
print("This is the important step.")
print()

policy.set_env(
    env
)


# dummy initialization 결과 제거
policy.reset_scenario_values()


print()
print("[4/4] Checking initialized solver...")
print()


assert policy.mpc_env is not None
assert policy.acados_solver is not None

assert policy.mpc_env.hum_model == "orca_casadi_kkt"
assert policy.human_pred_MID is True
assert policy.human_pred_MID_joint is True


print("====================================")
print("SICNAV / ACADOS READY")
print("====================================")

print(
    "humans       :",
    policy.num_hums
)

print(
    "horizon      :",
    policy.horiz
)

print(
    "dt           :",
    policy.time_step
)

print(
    "human model  :",
    policy.mpc_env.hum_model
)

print(
    "JMID enabled :",
    policy.human_pred_MID
)

print(
    "JMID joint   :",
    policy.human_pred_MID_joint
)

print(
    "nx           :",
    policy.mpc_env.nx
)

print(
    "nu           :",
    policy.mpc_env.nu
)

print(
    "robot inputs :",
    policy.mpc_env.nu_r
)

print(
    "MID samples  :",
    policy.mpc_env.num_MID_samples
)

print()
print("Acados solver initialization SUCCESS.")
print()
