from pathlib import Path
import numpy as np
import json


NB_INSTANCES = 10

# Links constants
NB_LINKS = 300
MAX_BANDWITH = 2
MAX_BINARY_FLOW = 2
MAX_SYMBOL_FLOW = 2
MAX_INV_BIN_FLOW_REQ = 0.7
MAX_GROUP_INV_BIN_FLOW = 0.8


def generate_all_instances(nb_instances, folder_path: Path):
    for i in range(nb_instances):
        links = []
        for _ in range(NB_LINKS):
            new_link = {}
            new_link["bandwidth"] = np.random.uniform(0, MAX_BANDWITH)
            new_link["binary_flow"] = np.random.uniform(0, MAX_BINARY_FLOW)
            new_link["symbol_flow"] = np.random.uniform(0, MAX_SYMBOL_FLOW)
            new_link["inverse_binary_flow"] = np.random.uniform(0, MAX_INV_BIN_FLOW_REQ)
            new_link["group_inverse_binary_flow"] = np.random.uniform(
                0, MAX_GROUP_INV_BIN_FLOW
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
