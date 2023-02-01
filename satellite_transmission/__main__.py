"""Run the optimization algorithm."""
import json
from pathlib import Path
from satellite_transmission.actor_critic import run_actor_critic


def main():
    with open(Path("./data/instances_100/instance2.json"), "r") as file:
        links = json.load(file)
    run_actor_critic(links, nb_episodes=1, duration_episode=50000)


if __name__ == "__main__":
    main()
