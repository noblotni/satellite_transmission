"""Contain configurable constants."""
from typing import Optional
from pathlib import Path


# Pytorch constants
# Number of threads Pytorch
# is allowed to use
TORCH_NB_THREADS: int = 2

# Actor-Critic constants
# Actor-critic random seed
AC_RANDOM_SEED: Optional[int] = None
# Report directory
AC_REPORT_DIR = Path("./output/actor_critic_results")
# Report directory when comparing PPO and A2C
AC_COMPARISON_REPORT_DIR = Path("./output/output/comparison/")
# Log directory
AC_LOG_DIR = Path("./output/actor_critic_logs")
# Name of the environment
ENV_NAME = "SatelliteRL"
