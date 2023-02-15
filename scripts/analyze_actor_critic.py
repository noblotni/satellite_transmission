"""Analyze actor-critic performances."""
import argparse
from pathlib import Path
import json
import pandas as pd
from satellite_transmission.actor_critic import run_actor_critic
from satellite_transmission.json_report import generate_solution_report


def main(args):
    with open(args.links_path, "r") as file:
        links = json.load(file)
    solutions = []
    for i in range(1):
        state_min, nb_mod, nb_grps = run_actor_critic(links, 5, 10000)
        generate_solution_report(
            state=state_min,
            links=links,
            output_path=Path("./state_min_{}.json".format(i)),
        )
        solutions.append((nb_mod, nb_grps))
    solutions_df = pd.DataFrame(solutions, columns=["nb_modems", "nb_groups"])
    solutions_df.to_csv("./solutions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("links_path", help="Path to the links data.", type=Path)
    args = parser.parse_args()
    main(args)
