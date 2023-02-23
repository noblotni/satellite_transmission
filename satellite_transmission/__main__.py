"""Run the optimization algorithm."""
import json
from pathlib import Path
import argparse
from satellite_transmission.actor_critic import run_actor_critic
from satellite_transmission.ppo import run_ppo
from satellite_transmission.json_report import generate_solution_report


def main(args):
    with open(args.links_path, "r") as file:
        links = json.load(file)
    if args.algo == "actor-critic":
        state_min, _, _ = run_actor_critic(
            links, nb_episodes=args.nb_episodes, duration_episode=args.duration_episode
        )
    elif args.algo == "ppo":
        state_min, _, _ = run_ppo(
            links, nb_episodes=args.nb_episodes, duration_episode=args.duration_episode
        )

    generate_solution_report(
        state=state_min, links=links, output_path=args.output_path
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
    parser.add_argument("--nb_episodes", help="Number of episodes.", type=int, default=10)
    parser.add_argument("--duration_episode", help="Duration of an episode.", type=int, default=10000)
    parser.add_argument("--output_path", "-o", help="Path to the output JSON file.", type=Path, default="./state_min.json")
    args = parser.parse_args()
    main(args)
