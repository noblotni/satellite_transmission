"""Generate instances."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

# Links constants
# Maximal bandwidth (kHz)
MAX_BANDWIDTH: int = 20000
# Minimal bandwidth (kHz)
MIN_BANDWIDTH: int = 10
# Maximal power of two for the binary rate
MAX_POW_BIN_RATE: int = 14
# Minimal power of two for the binary rate
MIN_POW_BIN_RATE: int = 5
# Maximal symbol rate (kbauds)
MAX_SYMBOL_RATE: int = 15000
# Minimal symbol rate (kbauds)
MIN_SYMBOL_RATE: int = 10
# Maximal binary rate in a group (kbps)
MAX_GROUP_INV_BIN_RATE: int = 32768


def get_date_prefix():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")


def generate_random_instances(nb_instances: int, nb_links: int, folder_path: Path) -> None:
    date_prefix = get_date_prefix()
    for i in range(nb_instances):
        links: list = []
        for _ in range(nb_links):
            bandwidth: float = np.random.uniform(MIN_BANDWIDTH, MAX_BANDWIDTH)
            binary_rate: int = 2 ** (
                np.random.randint(low=MIN_POW_BIN_RATE, high=MAX_POW_BIN_RATE + 1)
            )
            symbol_rate: float = np.random.uniform(MIN_SYMBOL_RATE, MAX_SYMBOL_RATE)
            group_inverse_inverse_binary_rate: float = np.random.uniform(
                binary_rate, MAX_GROUP_INV_BIN_RATE
            )
            links.append(
                {
                    "bandwidth": bandwidth,
                    "binary_rate": binary_rate,
                    "symbol_rate": symbol_rate,
                    "inverse_binary_rate": binary_rate,
                    "group_inverse_binary_rate": group_inverse_inverse_binary_rate,
                }
            )

        with open(f"{folder_path}/{date_prefix}_instance{i}.json", "w", encoding="utf-8") as file:
            json.dump(links, file, indent=4, sort_keys=True)


def generate_easy_instances(nb_instances: int, nb_links: int, folder_path: Path) -> None:
    date_prefix = get_date_prefix()
    for i in range(nb_instances):
        bandwidth: float = np.random.uniform(MIN_BANDWIDTH, MAX_BANDWIDTH)
        binary_rate: int = 2 ** (np.random.randint(low=MIN_POW_BIN_RATE, high=MAX_POW_BIN_RATE + 1))
        symbol_rate: float = np.random.uniform(MIN_SYMBOL_RATE, MAX_SYMBOL_RATE)
        group_inverse_inverse_binary_rate: float = np.random.uniform(
            binary_rate, MAX_GROUP_INV_BIN_RATE
        )
        links: list = [
            {
                "bandwidth": bandwidth,
                "binary_rate": binary_rate,
                "symbol_rate": symbol_rate,
                "inverse_binary_rate": binary_rate,
                "group_inverse_binary_rate": group_inverse_inverse_binary_rate,
            }
            for _ in range(nb_links)
        ]
        with open(f"{folder_path}/{date_prefix}_instance{i}.json", "w", encoding="utf-8") as file:
            json.dump(links, file, indent=4, sort_keys=True)


def main(args) -> None:
    folder_path = Path(f"./data/instances_{args.mode}_{args.nb_links}")
    folder_path.mkdir(parents=True, exist_ok=True)

    if args.mode == "easy":
        generate_easy_instances(
            nb_instances=args.nb_instances, nb_links=args.nb_links, folder_path=folder_path
        )
    elif args.mode == "random":
        generate_random_instances(
            nb_instances=args.nb_instances, nb_links=args.nb_links, folder_path=folder_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to generate instances.")
    parser.add_argument(
        "--nb-links", type=int, default=100, help="Number of links in the instance (default: 100)"
    )
    parser.add_argument(
        "--nb-instances", type=int, help="Number of instances to generate (default: 1)", default=1
    )
    parser.add_argument(
        "--type-instance",
        type=str,
        choices=["random", "easy"],
        default="random",
        help="Type of instance (default: random)",
        dest="mode",
    )
    args = parser.parse_args()
    main(args)
