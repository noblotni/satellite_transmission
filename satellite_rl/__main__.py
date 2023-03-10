"""Run the optimization algorithm."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import os
from termcolor import colored

from satellite_rl.comparison import batch_comparison
from satellite_rl.reinforcement_learning.agents.actor_critic import run_actor_critic
from satellite_rl.reinforcement_learning.agents.ppo import run_ppo
from satellite_rl.report import generate_solution_report, generate_report, generate_report_runs, generate_report_comparison


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Run the optimization algorithm."
    )
    parser.add_argument("links_path", help="Path to the links data.", type=Path)
    parser.add_argument(
        "--algo",
        help="Optimization algorithm used. (default: actor-critic)",
        type=str,
        default="actor-critic",
    )
    parser.add_argument("--nb_episodes", help="Number of episodes.", type=int,
                        default=10)
    parser.add_argument("--nb_timesteps", help="Duration of an episode.", type=int,
                        default=10000)
    parser.add_argument("--timeout", help="Time out (seconds).", type=int, 
                        default=0)
    parser.add_argument("--nb_repeat",
                        help="Number of times to repeat the optimization.", type=int,
                        default=1)
    parser.add_argument("--output_path", "-o", help="Path to the output JSON file.",
                    type=Path, default="satellite_rl/output/state_min.json")
    parser.add_argument("--verbose", help="Time out (seconds).", type=int, 
                        default=-1)
    parser.add_argument("--print_freq", help="Print frequency.", type=int, 
                        default=1000)
    parser.add_argument("--log_freq", help="Log frequency.", type=int, 
                        default=1000)
    parser.add_argument("--generate_report", choices=["True", "False"], help="Generate a report.", type=str, 
                        default="True")

    args: argparse.Namespace = parser.parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.links_path, "r") as file:
        links: list = json.load(file)

    filename = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    filename = filename.replace("/", "_")
    filename = filename.replace(":", "_")
    filename = filename.replace(" ", "_")
    verbose = args.verbose

    if verbose == -1:
        verbose = 0 if args.nb_repeat > 1 else 1

    generate_report_bool = True if args.generate_report == "True" else False

    print(
        f"Running {args.algo} algorithm..."
        if args.algo != "compare" else
        "Running Actor-Critic and PPO algorithms..."
    )
    print("=========================================")
    if args.nb_repeat > 1 or args.algo == "compare":
        print(f"Running {args.nb_repeat} times...")
        if args.algo == "compare":
            state_min, nb_grps_min, nb_mod_min, \
            results_dir_list_actor, results_dir_list_ppo = batch_comparison(links, args.algo, args.nb_episodes, 
                                                                            args.nb_timesteps, args.nb_repeat, 
                                                                            args.print_freq, args.log_freq,
                                                                            args.timeout, verbose, 
                                                                            generate_report_bool, filename)
        else:
            state_min, nb_grps_min, nb_mod_min, \
            report_paths = batch_comparison(links, args.algo, args.nb_episodes, 
                                            args.nb_timesteps, args.nb_repeat, 
                                            args.print_freq, args.log_freq, 
                                            args.timeout, verbose, 
                                            generate_report_bool, filename)
    elif args.nb_repeat == 1:
        if args.algo == "actor-critic":
            state_min, nb_grps_min, nb_mod_min, \
            report_path = run_actor_critic(links, args.nb_episodes, args.nb_timesteps, 
                                           args.print_freq, args.log_freq, args.timeout, 
                                           verbose, generate_report_bool, filename, False)
        elif args.algo == "ppo":
            state_min, nb_grps_min, nb_mod_min, \
            report_path = run_ppo(links, args.nb_episodes, args.nb_timesteps, 
                                  args.print_freq, args.log_freq, args.timeout, 
                                  verbose, generate_report_bool, filename, False)
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

    generate_solution_report(
        state=state_min, links=links, output_path=args.output_path
    )

    print("Solution saved in {}".format(args.output_path))
    print("=========================================")
    instance_name = str(args.links_path).split("/")[-1].split(".")[0]
    if generate_report_bool:
        if args.nb_repeat > 1 and args.algo != "compare":
            generate_report_runs(report_paths,
                        "/".join(report_paths[0].split("/")[:-2]) + "/report.html",
                        ("Actor-Critic Algorithm" if args.algo == "actor-critic" else "PPO Algorithm")+ f"\n{instance_name} - {args.nb_repeat} runs",
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
        elif args.algo == "compare":
            os.makedirs("comparison", exist_ok=True)
            generate_report_comparison(results_dir_list_actor, results_dir_list_ppo,
                        f"comparison/report_{args.nb_repeat}_runs_{datetime.now().strftime('%d/%m/%Y %H:%M:%S').replace('/','_').replace(':','_')}.html",
                        f"Actor-Critic vs PPO Algorithm\n{instance_name} - {args.nb_repeat} runs",
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
        else:
            generate_report(report_path + "time_step_report.csv",
                            report_path + "report.html",
                            ("Actor-Critic" if args.algo == "actor-critic" else "PPO") + \
                                f" Algorithm\n{instance_name} - {args.nb_repeat} runs",
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
    main()
