"""Run the optimization algorithm."""
import json
from pathlib import Path
from satellite_transmission.actor_critic import run_actor_critic


def main():
    with open(Path("./data/instances_100/instance0.json"), "r") as file:
        links = json.load(file)
    run_actor_critic(links, nb_episodes=50, duration_episode=2000)


if __name__ == "__main__":
    main()
