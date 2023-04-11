"""Contain configurable constants."""
import os
from pathlib import Path

# Pytorch constants
# Number of threads Pytorch
# is allowed to use
TORCH_NB_THREADS: int = int(os.environ.get("TORCH_NB_THREADS", default=2))

# Actor-Critic constants
# Actor-critic random seed
AC_RANDOM_SEED: int = int(os.environ.get("AC_RANDOM_SEED", default=0))
# Report directory
AC_REPORT_DIR = Path(
    os.environ.get("AC_REPORT_DIR", default="./satellite_rl/output/actor_critic_results")
)
# Report directory when comparing PPO and A2C
AC_COMPARISON_REPORT_DIR = Path(
    os.environ.get("AC_COMPARISON_REPORT_DIR", default="./satellite_rl/output/output/comparison/")
)
# Log directory
AC_LOG_DIR = Path(os.environ.get("AC_LOG_DIR", default="./satellite_rl/output/actor_critic_logs"))
# Name of the environment
ENV_NAME = os.environ.get("ENV_NAME", default="SatelliteRL")
