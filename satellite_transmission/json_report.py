"""Generate JSON report from a solution array."""
from pathlib import Path
import json
import numpy as np


def generate_solution_report(state: np.ndarray, links: list):
    """Generate a JSON report from the optimal state.

    Args:
        state (np.ndarray): optimal state returned by the optimization
            algorithm
        links (list): data about the satellite links
    """
    # Initialize the dictionary to
    # store in the JSON file
    json_report = {}
    for link_ind, coord in enumerate(state):
        group_key = "group_{}".format(coord[0])
        if not group_key in json_report.keys():
            json_report[group_key] = {
                "links": [],
                "nb_links": 0,
                "bandwidth_used": 0,
                "modems": {},
                "nb_modems": 0,
            }
        json_report[group_key]["links"].append(link_ind)
        json_report[group_key]["nb_links"] += 1
        json_report["banddwitdh_used"] += links[link_ind]["bandwidth"]
        json_report = generate_modem_report(
            json_report=json_report,
            modem_indice=coord[1],
            group_key=group_key,
            link_ind=link_ind,
            links=links,
        )
    with open(Path("./solution_report.json"), "w") as file:
        json.dump(json_report, file)


def generate_modem_report(
    json_report: dict, modem_indice: int, group_key: str, link_ind: int, links: list
):
    modem_key = "modem_{}".format(modem_indice)
    if not modem_key in json_report[group_key]["modems"].keys():
        json_report[group_key]["modems"][modem_key] = {
            "links": [],
            "binary_rate_used": 0,
            "symbol_rate_used": 0,
            "nb_links": 0,
        }
    json_report[group_key]["modems"][modem_key]["links"].append(link_ind)
    json_report[group_key]["modems"][modem_key]["links"]["nb_links"] += 1
    json_report[group_key]["modems"][modem_key]["binary_rate_used"] += links[link_ind][
        "binary_rate"
    ]
    json_report[group_key]["modems"][modem_key]["symbol_rate_used"] += links[link_ind][
        "symbol_rate"
    ]
    return json_report
