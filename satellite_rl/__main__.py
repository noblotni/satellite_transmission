"""Run the optimization algorithm."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from termcolor import colored
import pandas as pd
import subprocess
import webbrowser

from satellite_rl.comparison import batch_comparison
from satellite_rl.reinforcement_learning.agents.actor_critic import run_actor_critic
from satellite_rl.reinforcement_learning.agents.ppo import run_ppo
from satellite_rl.report import (
    generate_report,
    generate_report_comparison,
    generate_report_runs,
    generate_solution_report,
)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser("Run the optimization algorithm.")
    parser.add_argument("links_path", help="Path to the links data.", type=Path)
    parser.add_argument(
        "--algo",
        help="Optimization algorithm used. (default: actor-critic)",
        type=str,
        default="actor-critic",
    )
    parser.add_argument("--nb_episodes", help="Number of episodes.", type=int, default=10)
    parser.add_argument("--nb_timesteps", help="Duration of an episode.", type=int, default=10000)
    parser.add_argument("--timeout", help="Time out (seconds).", type=int, default=0)
    parser.add_argument(
        "--nb_repeat",
        help="Number of times to repeat the optimization.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output_path",
        "-o",
        help="Path to the output JSON file.",
        type=Path,
        default=Path("./state_min.json"),
    )
    parser.add_argument("--verbose", help="Verbosity.", type=int, default=-1)
    parser.add_argument("--print_freq", help="Print frequency.", type=int, default=1000)
    parser.add_argument("--log_freq", help="Log frequency.", type=int, default=1000)
    parser.add_argument(
        "--generate_report",
        choices=["True", "False"],
        help="Generate a report.",
        type=str,
        default="True",
    )

    args: argparse.Namespace = parser.parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.links_path, "r", encoding="utf-8") as file:
        links: list = json.load(file)

    now = datetime.now()
    filename = f"{args.algo}_{args.nb_repeat}_runs_{now.strftime('%Y')}-{now.strftime('%m')}-{now.strftime('%d')}-{now.strftime('%H')}-{now.strftime('%M')}-{now.strftime('%S')}"
    verbose = args.verbose

    if verbose == -1:
        verbose = 0 if args.nb_repeat > 1 else 1

    generate_report_bool = True if args.generate_report == "True" else False

    print(
        f"Running {args.algo} algorithm..."
        if args.algo != "compare"
        else "Running Actor-Critic and PPO algorithms..."
    )
    print("=========================================")
    if args.nb_repeat > 1 or args.algo == "compare":
        print(f"Running {args.nb_repeat} times...")
        if args.algo == "compare":
            state_min, nb_grps_min, nb_mod_min = batch_comparison(links, args.algo, args.nb_episodes, 
                                                                  args.nb_timesteps, args.nb_repeat, 
                                                                  args.print_freq, args.log_freq,
                                                                  args.timeout, verbose, 
                                                                  generate_report_bool, filename)
        else:
            state_min, nb_grps_min, nb_mod_min = batch_comparison(links, args.algo, args.nb_episodes, 
                                                                  args.nb_timesteps, args.nb_repeat, 
                                                                  args.print_freq, args.log_freq, 
                                                                  args.timeout, verbose, 
                                                                  generate_report_bool, filename)
    elif args.nb_repeat == 1:
        if args.algo == "actor-critic":
            state_min, nb_grps_min, nb_mod_min, \
            report_path = run_actor_critic(links, args.nb_episodes, args.nb_timesteps, 
                                           args.print_freq, args.log_freq, args.timeout, 
                                           verbose, generate_report_bool, filename, False, False)
        elif args.algo == "ppo":
            state_min, nb_grps_min, nb_mod_min, \
            report_path = run_ppo(links, args.nb_episodes, args.nb_timesteps, 
                                  args.print_freq, args.log_freq, args.timeout, 
                                  verbose, generate_report_bool, filename, False, False)
        else:
            raise ValueError("Unknown algorithm.")
        print("=========================================")
        print("RESULTS")
        print("=========================================")

    else:
        raise ValueError("nb_repeat must be positive.")

    print(colored("Best solution:", "blue"))
    print(f"nb_grps_min={colored(nb_grps_min, 'yellow')}")
    print(f"nb_mod_min={colored(nb_mod_min, 'yellow')}")

    print("=========================================")
    print("Saving the solution...")

    generate_solution_report(state=state_min, links=links, output_path=args.output_path)

    print("Solution saved in {}".format(args.output_path))
    print("=========================================")
    
    instance_name = os.path.basename(os.path.normpath(args.links_path)).split(".")[0]
    df_metadata = pd.DataFrame(
        {
            "Instance": [instance_name],
            "Number of episodes": [args.nb_episodes],
            "Number of timesteps": [args.nb_timesteps],
            "Number of runs": [args.nb_repeat],
            "Time out": [args.timeout],
            "Number of groups": [nb_grps_min],
            "Number of modems": [nb_mod_min],
            "Algorithm": [args.algo],
        }        
    )
    if generate_report_bool:
        if args.nb_repeat > 1 and args.algo != "compare":
            file_path = "satellite_rl/output/"
            file_path += "PPO_results/" if args.algo == "ppo" else "Actor-Critic_results/"
            metadata_path = file_path + f"SatelliteRL/{filename}/metadata.csv"
            file_path += f"SatelliteRL/{filename}/report.html"
            df_metadata.to_csv(metadata_path, index=False)
        elif args.algo == "compare":               
            os.makedirs("satellite_rl/output/comparison", exist_ok=True)
            file_path = f"satellite_rl/output/comparison/SatelliteRL/{filename}/report.html"
            df_metadata.to_csv(f"satellite_rl/output/comparison/SatelliteRL/{filename}/metadata.csv", index=False)
        else:
            df_metadata.to_csv(report_path + "metadata.csv", index=False)
        subprocess.call(["python", 
                         os.path.join(os.getcwd(), r"satellite_rl\report\report_dashboard.py")])
        webbrowser.open("http://127.0.0.1:8050/")


if __name__ == "__main__":
    main()
