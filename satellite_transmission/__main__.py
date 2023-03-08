"""Run the optimization algorithm."""
import json
from pathlib import Path
import argparse
from satellite_transmission.actor_critic import run_actor_critic
from satellite_transmission.ppo import run_ppo
from satellite_transmission.json_report import generate_solution_report
import numpy as np
from termcolor import colored
from satellite_transmission.batch_comparison import batch_comparison
from satellite_transmission.report_generation import generate_report
from datetime import datetime

def main(args):
    with open(args.links_path, "r") as file:
        links = json.load(file)

    verbose = args.verbose
    if verbose == -1:
        verbose = 0 if args.nb_repeat > 1 else 1

    print(f"Running {args.algo} algorithm...") if args.algo != "compare" else print("Running Actor-Critic and PPO algorithms...")
    print("=========================================")
    if args.nb_repeat > 1 or args.algo == "compare":
        print(f"Running {args.nb_repeat} times...")
        state_min, nb_grps_min, nb_mod_min = batch_comparison(links, args.algo, args.nb_episodes, args.nb_timesteps, args.print_freq, args.log_freq, args.nb_repeat, args.timeout, verbose, args.generate_report)
    elif args.nb_repeat == 1:
        if args.algo == "actor-critic":
            state_min, nb_grps_min, nb_mod_min, report_path = run_actor_critic(links, args.nb_episodes, args.nb_timesteps, args.print_freq, args.log_freq, args.timeout, verbose, args.generate_report)
        elif args.algo == "ppo":
            state_min, nb_grps_min, nb_mod_min, report_path = run_ppo(links, args.nb_episodes, args.nb_timesteps, args.print_freq, args.log_freq, args.timeout, verbose, args.generate_report)
        else:
            raise ValueError("Unknown algorithm.")
        print("=========================================")
        print("RESULTS")
        print("=========================================")
    else:
        raise ValueError("nb_repeat must be positive.")
    print(colored("Best solution:","blue"))
    print(f"nb_grps_min={colored(nb_grps_min, 'yellow')}")
    print(f"nb_mod_min={colored(nb_mod_min, 'yellow')}")

    print("=========================================")
    print("Saving the solution...")
        
    generate_solution_report(
        state=state_min, links=links, output_path=args.output_path
    )
    if args.generate_report:
        generate_report(report_path + "time_step_report.csv",
                        report_path + "report.html",
                        "Actor-Critic" if args.algo == "actor-critic" else "PPO" + " Algorithm",
                        {
                            "Number of episodes": args.nb_episodes,
                            "Number of timesteps": args.nb_timesteps,
                            "Number of runs": args.nb_repeat,
                            "Time out": args.timeout,
                            "Number of groups": nb_grps_min,
                            "Number of modems": nb_mod_min,
                            "Time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                            "Number of links": len(links)
                        }
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
    parser.add_argument("--nb_timesteps", help="Number of timesteps", type=int, default=10000)
    parser.add_argument("--timeout", help="Time out (seconds).", type=int, default=0)
    parser.add_argument("--nb_repeat", help="Number of times to repeat the optimization.", type=int, default=1)

    parser.add_argument("--output_path", "-o", help="Path to the output JSON file.", type=Path, default="./state_min.json")
    parser.add_argument("--verbose", help="Time out (seconds).", type=int, default=-1)
    parser.add_argument("--print_freq", help="Print frequency.", type=int, default=1000)
    parser.add_argument("--log_freq", help="Log frequency.", type=int, default=1000)
    parser.add_argument("--generate_report", help="Generate a report.", type=bool, default=True)

    args = parser.parse_args()
    main(args)
