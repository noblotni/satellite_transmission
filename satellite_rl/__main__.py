"""Run the optimization algorithm."""
import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
from termcolor import colored

from satellite_rl.comparison import batch_comparison
from satellite_rl.reinforcement_learning.agents.actor_critic import run_actor_critic
from satellite_rl.reinforcement_learning.agents.ppo import run_ppo
from satellite_rl.reinforcement_learning.environment import solve_easy_instances
from satellite_rl.report import generate_solution_report, launch_server


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser("Run the optimization algorithm.")
    parser.add_argument("links_path", help="Path to the links data.", type=Path)
    parser.add_argument(
        "--algo",
        help="Optimization algorithm used. (default: actor-critic)",
        choices=["actor-critic", "ppo", "compare"],
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
    parser.add_argument(
        "--easy_instances",
        choices=["True", "False"],
        help="Solve easy instances.",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--type", help="Train or evaluate.", type=str, choices=["train", "eval"], default="train"
    )

    if parser.parse_args().type == "train":
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
                state_min, nb_grps_min, nb_mod_min = batch_comparison(
                    links,
                    args.algo,
                    args.nb_episodes,
                    args.nb_timesteps,
                    args.nb_repeat,
                    args.print_freq,
                    args.log_freq,
                    args.timeout,
                    verbose,
                    generate_report_bool,
                    filename,
                )
            else:
                state_min, nb_grps_min, nb_mod_min = batch_comparison(
                    links,
                    args.algo,
                    args.nb_episodes,
                    args.nb_timesteps,
                    args.nb_repeat,
                    args.print_freq,
                    args.log_freq,
                    args.timeout,
                    verbose,
                    generate_report_bool,
                    filename,
                )
        elif args.nb_repeat == 1:
            if args.algo == "actor-critic":
                state_min, nb_grps_min, nb_mod_min = run_actor_critic(
                    links=links,
                    nb_episodes=args.nb_episodes,
                    duration_episode=args.nb_timesteps,
                    print_freq=args.print_freq,
                    log_freq=args.log_freq,
                    timeout=args.timeout,
                    verbose=verbose,
                    report=generate_report_bool,
                    filename=filename,
                    batch=False,
                    compare=False,
                )
            elif args.algo == "ppo":
                state_min, nb_grps_min, nb_mod_min = run_ppo(
                    links=links,
                    nb_episodes=args.nb_episodes,
                    nb_timesteps=args.nb_timesteps,
                    print_freq=args.print_freq,
                    log_freq=args.log_freq,
                    timeout=args.timeout,
                    verbose=verbose,
                    report=generate_report_bool,
                    filename=filename,
                    batch=False,
                    compare=False,
                )
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

        instance_name = args.links_path.stem
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
            if args.easy_instances == "True":
                nb_mod_opt, nb_grps_opt = solve_easy_instances(links)
                df_opt = pd.DataFrame(
                    {
                        "Instance": [instance_name],
                        "Number of modems": [nb_mod_opt],
                        "Number of groups": [nb_grps_opt],
                    }
                )
            if args.algo != "compare":
                metadata_dir = Path("satellite_rl/output/")
                metadata_dir /= "PPO_results/" if args.algo == "ppo" else "actor_critic_results/"
                metadata_dir /= f"SatelliteRL/{filename}"
                metadata_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = metadata_dir / "metadata.csv"
                df_metadata.to_csv(metadata_path, index=False)
                opt_path = metadata_dir / "optimal_solution.csv"
                if args.easy_instances == "True":
                    df_opt.to_csv(opt_path, index=False)
            else:
                metadata_dir = Path(f"satellite_rl/output/comparison/SatelliteRL/{filename}")
                metadata_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = metadata_dir / "metadata.csv"
                df_metadata.to_csv(metadata_path, index=False)
                opt_path = metadata_dir / "optimal_solution.csv"
                if args.easy_instances == "True":
                    df_opt.to_csv(opt_path, index=False)

            is_launched = launch_server()
            if is_launched:
                webbrowser.open("http://localhost:8050")
    else:
        is_launched = launch_server()
        if is_launched:
            print("Server launched.")
            print("Open http://localhost:8050 in your browser.")
            print("=========================================")
            webbrowser.open("http://localhost:8050")
        else:
            print("Server already launched.")
            print("Open http://localhost:8050 in your browser.")
            print("=========================================")
            webbrowser.open("http://localhost:8050")


if __name__ == "__main__":
    main()
