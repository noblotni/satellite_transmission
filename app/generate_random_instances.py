"""Generate random instances."""
import json
from pathlib import Path

import numpy as np

NB_INSTANCES: int = 10

# Links constants
NB_LINKS: int = 100
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


def generate_all_instances(nb_instances: int, folder_path: Path) -> None:
    for i in range(nb_instances):
        links: list = []
        for _ in range(NB_LINKS):
            bandwidth: float = np.random.uniform(MIN_BANDWIDTH, MAX_BANDWIDTH)
            binary_rate: int = 2 ** (
                np.random.randint(low=MIN_POW_BIN_RATE, high=MAX_POW_BIN_RATE + 1)
            )
            symbol_rate: float = np.random.uniform(MIN_SYMBOL_RATE, MAX_SYMBOL_RATE)
            group_inverse_inverse_binary_rate: float = np.random.uniform(
                binary_rate, MAX_GROUP_INV_BIN_RATE
            )
            links.append({
                "bandwidth": bandwidth,
                "binary_rate": binary_rate,
                "symbol_rate": symbol_rate,
                "inverse_binary_rate": binary_rate,
                "group_inverse_binary_rate": group_inverse_inverse_binary_rate,
            })

        with open(f"{folder_path}/instance{i}.json", "w") as file:
            json.dump(links, file, indent=4, sort_keys=True)


def main() -> None:
    folder_path = Path(f"app/data/instances_random_{NB_LINKS}")
    if not (folder_path.exists()):
        folder_path.mkdir()

    generate_all_instances(nb_instances=NB_INSTANCES, folder_path=folder_path)
