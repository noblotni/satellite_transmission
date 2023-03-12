"""Generate JSON report from a solution array."""
import json
from pathlib import Path

import numpy as np


def generate_solution_report(state: np.ndarray, links: list, output_path: Path) -> None:
    """Generate a JSON report from the optimal state.
    Args:
        state (np.ndarray): optimal state returned by the optimization
            algorithm
        links (list): data about the satellite links
        output_path (Path): location of the output JSON file
    """
    # Initialize the dictionary to
    # store in the JSON file
    json_report: dict = {"groups": {}, "nb_groups": 0, "nb_modems": 0}
    for link_ind, coord in enumerate(state):
        group_key = f"group_{coord[0]}"
        if not group_key in json_report["groups"].keys():
            json_report["groups"][group_key]: dict = {
                "links": [],
                "nb_links": 0,
                "bandwidth_used": 0,
                "modems": {},
                "nb_modems": 0,
            }
        json_report["groups"][group_key]["links"].append(link_ind)
        json_report["groups"][group_key]["nb_links"] += 1
        json_report["groups"][group_key]["bandwidth_used"] += links[link_ind]["bandwidth"]
        json_report: dict = generate_modem_report(
            json_report=json_report,
            modem_indice=coord[1],
            group_key=group_key,
            link_ind=link_ind,
            links=links,
        )
        json_report["groups"][group_key]["nb_modems"] = len(
            json_report["groups"][group_key]["modems"]
        )
    json_report["nb_groups"]: int = len(json_report["groups"].keys())
    json_report["nb_modems"]: int = sum(
        [json_report["groups"][group]["nb_modems"] for group in json_report["groups"]]
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(json_report, file, indent=4, sort_keys=True)


def generate_modem_report(
    json_report: dict, modem_indice: int, group_key: str, link_ind: int, links: list
) -> dict:
    modem_key: str = "modem_{}".format(modem_indice)
    if modem_key not in json_report["groups"][group_key]["modems"].keys():
        json_report["groups"][group_key]["modems"][modem_key]: dict = {
            "links": [],
            "binary_rate_used": 0,
            "symbol_rate_used": 0,
            "nb_links": 0,
        }
    json_report["groups"][group_key]["modems"][modem_key]["links"].append(link_ind)
    json_report["groups"][group_key]["modems"][modem_key]["nb_links"] += 1
    json_report["groups"][group_key]["modems"][modem_key]["binary_rate_used"] += links[link_ind][
        "binary_rate"
    ]
    json_report["groups"][group_key]["modems"][modem_key]["symbol_rate_used"] += links[link_ind][
        "symbol_rate"
    ]
    return json_report
