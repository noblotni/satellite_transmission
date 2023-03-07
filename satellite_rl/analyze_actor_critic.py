"""Analyze actor-critic performances."""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch

from satellite_rl.reinforcement_learning.agents.actor_critic import run_actor_critic
from satellite_rl.report import generate_solution_report


def solve(links: list, i: int) -> tuple[int, int]:
    # Set a new random seed
    torch.random.seed()
    state_min, nb_mod, nb_grps = run_actor_critic(links, 5, 10000)
    generate_solution_report(
        state=state_min,
        links=links,
        output_path=Path("satellite_rl/output/state_min_{}.json".format(i)),
    )
    return nb_mod, nb_grps


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("links_path", help="Path to the links data.", type=Path)
    args: argparse.Namespace = parser.parse_args()

    with open(args.links_path, "r") as file:
        links: list = json.load(file)

    with Pool(processes=2) as pool:
        solutions: list = pool.starmap(solve, [(links, i) for i in range(30)])
    solutions_df: pd.DataFrame = pd.DataFrame(
        solutions, columns=["nb_modems", "nb_groups"]
    )
    solutions_df.to_csv("satellite_rl/output/solutions.csv")
