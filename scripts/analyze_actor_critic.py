"""Analyze actor-critic performances."""
import argparse
from pathlib import Path
import json
import pandas as pd
from satellite_transmission.actor_critic import run_actor_critic
import numpy as np


def main(args):
    with open(args.links_path, "r") as file:
        links = json.load(file)
    solutions = []
    for i in range(10):
        state_min, nb_mod, nb_grps = run_actor_critic(links, 1, 20000)
        np.savez_compressed("./state_min_{}.npz".format(i), state_min)
        solutions.append((nb_mod, nb_grps))
    solutions_df = pd.DataFrame(solutions, columns=["nb_modems", "nb_groups"])
    solutions_df.to_csv("./solutions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("links_path", help="Path to the links data.", type=Path)
    args = parser.parse_args()
    main(args)
