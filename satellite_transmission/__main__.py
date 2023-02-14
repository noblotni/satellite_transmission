"""Run the optimization algorithm."""
import json
from pathlib import Path
import argparse
from satellite_transmission.actor_critic import run_actor_critic
from satellite_transmission.json_report import generate_solution_report


def main(args):
    with open(args.links_path, "r") as file:
        links = json.load(file)
    if args.algo == "actor-critic":
        state_min, _, _ = run_actor_critic(links, nb_episodes=1, duration_episode=30000)
    generate_solution_report(
        state=state_min, links=links, output_path=Path("./state_min.json")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the optimization algorithm.")
    parser.add_argument("links_path", help="Path to the links data.", type=Path)
    parser.add_argument(
        "--algo",
        help="Optimization algorithm used. (default: actor-critic)",
        type=str,
        default="actor-critic",
    )
    args = parser.parse_args()
    main(args)
