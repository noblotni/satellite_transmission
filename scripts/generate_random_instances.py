"""Generate random instances."""
from pathlib import Path
import numpy as np
import json


NB_INSTANCES = 10

# Links constants
NB_LINKS = 50
# Maximal bandwidth (kHz)
MAX_BANDWIDTH = 20000
# Minimal bandwidth (kHz)
MIN_BANDWIDTH = 10
# Maximal power of two for the binary rate
MAX_POW_BIN_RATE = 14
# Minimal power of two for the binary rate
MIN_POW_BIN_RATE = 5
# Maximal symbol rate (kbauds)
MAX_SYMBOL_RATE = 15000
# Minimal symbol rate (kbauds)
MIN_SYMBOL_RATE = 10
# Maximal binary rate in a group (kbps)
MAX_GROUP_INV_BIN_RATE = 32768


def generate_all_instances(nb_instances, folder_path: Path):
    for i in range(nb_instances):
        links = []
        for _ in range(NB_LINKS):
            new_link = {}
            new_link["bandwidth"] = np.random.uniform(MIN_BANDWIDTH, MAX_BANDWIDTH)
            new_link["binary_rate"] = 2 ** (
                np.random.randint(low=MIN_POW_BIN_RATE, high=MAX_POW_BIN_RATE + 1)
            )
            new_link["symbol_rate"] = np.random.uniform(
                MIN_SYMBOL_RATE, MAX_SYMBOL_RATE
            )
            new_link["inverse_binary_rate"] = new_link["binary_rate"]
            new_link["group_inverse_binary_rate"] = np.random.uniform(
                new_link["inverse_binary_rate"], MAX_GROUP_INV_BIN_RATE
            )
            links.append(new_link)
            with open(folder_path / ("instance" + str(i) + ".json"), "w") as file:
                json.dump(links, file)


def main():

    folder_path = Path("./instances_" + str(NB_LINKS))
    if not (folder_path.exists()):
        folder_path.mkdir()

    generate_all_instances(nb_instances=NB_INSTANCES, folder_path=folder_path)


if __name__ == "__main__":
    main()
