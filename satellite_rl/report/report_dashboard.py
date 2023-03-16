from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State

OUTPUT_PATH = Path.cwd() / "satellite_rl" / "output"
ALL_ACTOR_CRITIC_RUN_PATHS = OUTPUT_PATH / "actor-critic_Results" / "SatelliteRL"
ALL_PPO_RUN_PATHS = OUTPUT_PATH / "PPO_Results" / "SatelliteRL"
ALL_COMPARISON_RUN_PATHS = OUTPUT_PATH / "comparison" / "SatelliteRL"


def get_run_paths():
    all_actor_critic_runs_path = list(ALL_ACTOR_CRITIC_RUN_PATHS.glob("*"))
    all_ppo_runs_path = list(ALL_PPO_RUN_PATHS.glob("*"))
    all_comparison_runs_path = list(ALL_COMPARISON_RUN_PATHS.glob("*"))

    all_runs_path = all_actor_critic_runs_path + all_ppo_runs_path + all_comparison_runs_path
    all_runs_path = sorted(all_runs_path, key=lambda chemin: chemin.stat().st_ctime)
    options = []
    for path in all_runs_path:
        options.append({"label": path.name, "value": str(path)})

    all_runs_path_witout_comp = all_actor_critic_runs_path + all_ppo_runs_path
    all_runs_path_witout_comp = sorted(
        all_runs_path_witout_comp, key=lambda chemin: chemin.stat().st_ctime
    )
    option_witout_comp = []
    for path in all_runs_path_witout_comp:
        option_witout_comp.append({"label": path.name, "value": str(path)})
    if len(option_witout_comp) == 0:
        option_witout_comp = options
    return options, option_witout_comp


options, option_witout_comp = get_run_paths()

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("Report dashboard"),
                html.Label("Select an option:", className="run-text-title"),
                dcc.RadioItems(
                    id="option-selector",
                    options=[
                        {"label": "Compare two runs", "value": "compare"},
                        {"label": "View a run", "value": "view"},
                    ],
                    value="view",
                    className="run-text",
                ),
                html.Div(
                    [
                        html.Label("Select a run:", className="run-text-title"),
                        dcc.Dropdown(
                            id="run-selector",
                            options=options,
                            value=options[0]["value"],
                            searchable=False,
                            clearable=False,
                            className="run-dropdown",
                        ),
                    ],
                    id="run-selector-container",
                ),
                html.Div(
                    [
                        html.Label("Select run A:", className="run-text-title"),
                        dcc.Dropdown(
                            id="run-a-selector",
                            options=option_witout_comp,
                            value=option_witout_comp[0]["value"],
                            searchable=False,
                            clearable=False,
                            className="run-dropdown",
                        ),
                    ],
                    id="run-a-selector-container",
                    style={"display": "none"},
                ),
                html.Div(
                    [
                        html.Label("Select run B:", className="run-text-title"),
                        dcc.Dropdown(
                            id="run-b-selector",
                            options=option_witout_comp,
                            value=option_witout_comp[0]["value"],
                            searchable=False,
                            clearable=False,
                            className="run-dropdown",
                        ),
                    ],
                    id="run-b-selector-container",
                    style={"display": "none"},
                ),
            ],
            className="column left",
        ),
        html.Div(
            [
                html.H1("Report dashboard"),
                html.Div(
                    [
                        dash_table.DataTable(
                            id="table",
                            columns=[
                                {"name": i, "id": i}
                                for i in [
                                    "Instance",
                                    "Number of episodes",
                                    "Number of timesteps",
                                    "Number of runs",
                                    "Time out",
                                    "Number of groups",
                                    "Number of modems",
                                    "Algorithm",
                                ]
                            ],
                            data=[],
                            style_cell_conditional=[
                                {"if": {"column_id": "Region"}, "textAlign": "center"}
                            ],
                            style_cell={"padding": "5px", "textAlign": "center"},
                            style_header={
                                "backgroundColor": "black",
                                "fontWeight": "bold",
                                "color": "white",
                            },
                            style_data={
                                "whiteSpace": "normal",
                                "backgroundColor": "white",
                            },
                        )
                    ]
                ),
                html.Div(
                    [dcc.Graph(id="graph_boxplot", className="graph")],
                    className="row",
                    style={"display": "none"},
                    id="graph_boxplot_container",
                ),
                html.Div([dcc.Graph(id="graph_episode", className="graph")], className="row"),
                html.Div([dcc.Graph(id="graph_timestep", className="graph")], className="row"),
            ],
            className="column right",
        ),
    ],
    className="container",
)


@app.callback(
    [
        Output("run-selector-container", "style"),
        Output("run-a-selector-container", "style"),
        Output("run-b-selector-container", "style"),
    ],
    [Input("option-selector", "value")],
)
def toggle_run_selector(option):
    """
    Toggle the run selector.

    This function controls the visibility of the different run selectors based on the selected option.

    Args:
        option (str): The selected option.

    Returns:
        dict: A dictionary containing the styles for the run selectors.
    """
    if option == "compare":
        return {"display": "none"}, {"display": "block"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "none"}, {"display": "none"}


def load_metadata(path):
    """
    Load metadata from a CSV file located in a specified path.

    This function loads a metadata file in CSV format located at the given path and returns it as a pandas DataFrame.

    Args:
        path (str or Path): The path to the directory containing the metadata.csv file.

    Returns:
        metadata (pandas.DataFrame): The metadata contained in the metadata.csv file as a DataFrame.
    """
    path = Path(path)
    metadata_file = path / "metadata.csv"
    metadata = pd.read_csv(metadata_file, index_col=0)
    return metadata


def load_data(run, algo):
    """
    Load the data from the specified run and algorithm.

    Args:
        run (str): The path to the run directory.
        algo (str): The algorithm to load the data for. Must be one of "actor-critic", "PPO", or "compare".

    Returns:
        If `algo` is not "compare", a list of pandas dataframes loaded from the "report*.csv" files in the specified run directory.
        If `algo` is "compare", a tuple of two lists of pandas dataframes, one for each algorithm ("actor-critic" and "PPO"), loaded from their respective "report*.csv" files in the specified run directory.
    """
    if algo != "compare":
        dfs = [
            pd.read_csv(path).reset_index(drop=True) for path in list(Path(run).glob("report*.csv"))
        ]
        return dfs
    else:
        dfs_actor = [
            pd.read_csv(path).reset_index(drop=True)
            for path in list(Path(run, "actor-critic").glob("report*.csv"))
        ]

        dfs_ppo = [
            pd.read_csv(path).reset_index(drop=True)
            for path in list(Path(run, "PPO").glob("report*.csv"))
        ]
        return dfs_actor, dfs_ppo


def get_episode_df(dfs):
    """
    Groups the input DataFrames by episode and creates new DataFrames containing
    information about the episode's reward, minimum number of modems used, and
    minimum number of groups used.

    Args:
        dfs (list of pd.DataFrame): List of pandas DataFrames.

    Returns:
        df_episodes (list of pd.DataFrame): List of pandas DataFrames where each
        DataFrame contains information about the episode's reward, minimum number
        of modems used, and minimum number of groups used.
    """
    df_episode_reward = [df.groupby("episode")["reward"].max() for df in dfs]
    df_episode_modem = [df.groupby("episode")["nb_modem_min"].min() for df in dfs]
    df_episode_group = [df.groupby("episode")["nb_group_min"].min() for df in dfs]
    df_episodes = [
        pd.concat([df_ep_reward, df_ep_modem, df_ep_group], axis=1)
        for df_ep_reward, df_ep_modem, df_ep_group in zip(
            df_episode_reward, df_episode_modem, df_episode_group
        )
    ]
    return df_episodes


def get_compare_df(dfs_actor, dfs_ppo):
    """
    Returns dataframes containing the best and worst performing episodes for the actor-critic and PPO models.

    Args:
        dfs_actor (list of pandas.DataFrame): A list of dataframes containing the results of the actor-critic model.
        dfs_ppo (list of pandas.DataFrame): A list of dataframes containing the results of the PPO model.

    Returns:
        best_df_actor (pandas.DataFrame): A dataframe containing the best performing episode for the actor-critic model.
        best_df_ppo (pandas.DataFrame): A dataframe containing the best performing episode for the PPO model.
        worst_df_actor (pandas.DataFrame): A dataframe containing the worst performing episode for the actor-critic model.
        worst_df_ppo (pandas.DataFrame): A dataframe containing the worst performing episode for the PPO model.
    """
    best_df_actor_index = np.argmin(
        [
            df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min()
            for df in dfs_actor
        ]
    )
    best_df_actor = dfs_actor[best_df_actor_index]
    best_df_ppo_index = np.argmin(
        [df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs_ppo]
    )
    best_df_ppo = dfs_ppo[best_df_ppo_index]
    worst_df_actor_index = np.argmax(
        [
            df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min()
            for df in dfs_actor
        ]
    )
    worst_df_actor = dfs_actor[worst_df_actor_index]
    worst_df_ppo_index = np.argmax(
        [df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs_ppo]
    )
    worst_df_ppo = dfs_ppo[worst_df_ppo_index]
    return best_df_actor, best_df_ppo, worst_df_actor, worst_df_ppo


def get_scatter_modem_group_reward(dfs, title, names=None):
    """
    Create a scatter plot of the minimum number of modems and groups per episode and the reward per episode for each dataframe in dfs.

    Args:
        dfs (list): A list of pandas dataframes containing data for each episode.
        title (str): The title of the plot.
        names (list, optional): A list of names to label each dataframe in dfs. Defaults to None.

    Returns:
        fig (plotly.graph_objs._figure.Figure): The scatter plot figure.
    """
    if names is None:
        names = range(1, len(dfs) + 1)
    fig = go.Figure()
    for name, df in zip(names, dfs):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["nb_modem_min"],
                name="nb_modem_min_" + str(name),
                mode="lines",
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["nb_group_min"],
                name="nb_group_min_" + str(name),
                mode="lines",
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["reward"], name="Reward_" + str(name), mode="lines", yaxis="y2"
            )
        )
    fig.update_layout(
        title={
            "text": title,
            "y": 0.9,  # new
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",  # new
        },
        xaxis=dict(title=title),
        yaxis=dict(title="Nombre de modem et de groupes"),
        yaxis2=dict(
            title="Reward",
            overlaying="y",
            side="right",
        ),
        legend={"xanchor": "right", "x": 1.4},
    )
    return fig


def get_box_modem_group_reward(dfs_best, title, names):
    """
    Create a box plot figure to visualize the number of modems and groups, as well as the reward, for different runs.

    Args:
        dfs_best (dict): A dictionary containing the best dataframes for each run. Keys are run names and values are dataframes.
        title (str): The title of the figure.
        names (list): A list of run names to include in the figure.

    Returns:
        fig (plotly.graph_objs.Figure): A box plot figure with the specified data and layout.
    """
    fig = go.Figure()
    for name, y_axis in zip(names, ["y1", "y2", "y3", "y4"]):
        fig.add_trace(go.Box(y=dfs_best[name], name=name, yaxis=y_axis))
    fig.update_layout(
        title={"text": title, "y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        xaxis=dict(title="Timestep"),
        yaxis=dict(title="Nombre de modem et de groupes"),
        yaxis2=dict(overlaying="y", anchor="free", autoshift=True, title="Modem"),
        yaxis3=dict(overlaying="y", anchor="free", autoshift=True, title="Group"),
        yaxis4=dict(overlaying="y", side="right", title="Reward"),
        legend_title="Runs",
        legend={"xanchor": "right", "x": 1.4},
    )
    return fig


@app.callback(
    [
        Output("table", "data"),
        Output("graph_boxplot_container", "style"),
        Output("graph_episode", "figure"),
        Output("graph_timestep", "figure"),
        Output("graph_boxplot", "figure"),
    ],
    [
        Input("run-selector", "value"),
        Input("run-a-selector", "value"),
        Input("run-b-selector", "value"),
    ],
    [State("option-selector", "value")],
)
def update_graphs(run, run_a, run_b, option):
    """
    Update graphs according to selected options.

    Args:
        run (str): The current run name.
        run_a (str): The first run name to compare.
        run_b (str): The second run name to compare.
        option (str): The option selected on the app.

    Returns:
        list of dicts: The updated table of metadata.
        dict: The updated style of the boxplot container graph.
        dict: The updated scatterplot of the episode reward.
        dict: The updated scatterplot of the timestep reward.
        dict: The updated boxplot of the best modem and group count.
    """
    if option == "compare":
        metadata_a = load_metadata(run_a).reset_index(drop=False)
        metadata_b = load_metadata(run_b).reset_index(drop=False)
        nb_runs_a = metadata_a["Number of runs"].iloc[0]
        nb_runs_b = metadata_b["Number of runs"].iloc[0]
        algo_a = metadata_a["Algorithm"].iloc[0]
        algo_b = metadata_b["Algorithm"].iloc[0]

        style_box = (
            {"display": "block"} if (nb_runs_a > 1) & (nb_runs_b > 1) else {"display": "none"}
        )

        metadata = pd.concat([metadata_a, metadata_b])
        table = metadata.to_dict("records")

        dfs_a = load_data(run_a, algo_a)
        dfs_b = load_data(run_b, algo_b)
        df_episodes_a = get_episode_df(dfs_a)
        df_episodes_b = get_episode_df(dfs_b)
        dfs = dfs_a + dfs_b
        df_episodes = df_episodes_a + df_episodes_b

        fig1 = get_scatter_modem_group_reward(
            df_episodes,
            "Episode",
            names=["algo_a_" + str(i) for i in range(1, nb_runs_a + 1)]
            + ["algo_b_" + str(i) for i in range(1, nb_runs_b + 1)],
        )
        fig2 = get_scatter_modem_group_reward(
            dfs,
            "Timestep",
            names=["algo_a_" + str(i) for i in range(1, nb_runs_a + 1)]
            + ["algo_b_" + str(i) for i in range(1, nb_runs_b + 1)],
        )

        if (nb_runs_a > 1) and (nb_runs_b > 1):
            dfs_modems_min_a = [df["nb_modem_min"].min() for df in dfs_a]
            dfs_groups_min_a = [df["nb_group_min"].min() for df in dfs_a]
            dfs_modems_min_b = [df["nb_modem_min"].min() for df in dfs_b]
            dfs_groups_min_b = [df["nb_group_min"].min() for df in dfs_b]

            dfs_best = pd.DataFrame(
                {
                    "Modem_a": dfs_modems_min_a,
                    "Modem_b": dfs_groups_min_a,
                    "Group_a": dfs_modems_min_b,
                    "Group_b": dfs_groups_min_b,
                }
            )
            fig3 = get_box_modem_group_reward(
                dfs_best, "Boxplot", ["Modem_a", "Modem_b", "Group_a", "Group_b"]
            )
        else:
            fig3 = go.Figure()
    else:
        metadata = load_metadata(run).reset_index(drop=False)
        algo = metadata["Algorithm"].iloc[0]
        nb_runs = metadata["Number of runs"].iloc[0]
        style_box = {"display": "block"} if (nb_runs > 1) else {"display": "none"}
        table = metadata.to_dict("records")
        if algo != "compare":
            dfs = load_data(run, algo)
            df_episodes = get_episode_df(dfs)

            fig1 = get_scatter_modem_group_reward(df_episodes, "Episode")
            fig2 = get_scatter_modem_group_reward(dfs, "Timestep")

            dfs_add_min = [
                df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min()
                for df in dfs
            ]
            dfs_modems_min = [df["nb_modem_min"].min() for df in dfs]
            dfs_groups_min = [df["nb_group_min"].min() for df in dfs]
            dfs_reward_min = [df["reward"].min() for df in dfs]
            dfs_best = pd.DataFrame(
                {
                    "Modem_Group": dfs_add_min,
                    "Modem": dfs_modems_min,
                    "Group": dfs_groups_min,
                    "Reward": dfs_reward_min,
                }
            )
            fig3 = get_box_modem_group_reward(
                dfs_best, "Boxplot", ["Modem_Group", "Modem", "Group", "Reward"]
            )

        else:
            dfs_actor, dfs_ppo = load_data(run, algo)
            best_df_actor, best_df_ppo, worst_df_actor, worst_df_ppo = get_compare_df(
                dfs_actor, dfs_ppo
            )
            best_df_actor_ep = get_episode_df([best_df_actor])[0]
            best_df_ppo_ep = get_episode_df([best_df_ppo])[0]
            worst_df_actor_ep = get_episode_df([worst_df_actor])[0]
            worst_df_ppo_ep = get_episode_df([worst_df_ppo])[0]
            dfs = [best_df_actor, best_df_ppo, worst_df_actor, worst_df_ppo]
            dfs_ep = [best_df_actor_ep, best_df_ppo_ep, worst_df_actor_ep, worst_df_ppo_ep]

            fig1 = get_scatter_modem_group_reward(
                dfs_ep, "Episode", names=["Best Actor", "Best PPO", "Worst Actor", "Worst PPO"]
            )
            fig2 = get_scatter_modem_group_reward(
                dfs, "Timestep", names=["Best Actor", "Best PPO", "Worst Actor", "Worst PPO"]
            )

            dfs_best = pd.DataFrame(
                {
                    "Modem_actor": [df["nb_modem_min"].min() for df in dfs_actor],
                    "Group_actor": [df["nb_group_min"].min() for df in dfs_actor],
                    "Modem_ppo": [df["nb_modem_min"].min() for df in dfs_ppo],
                    "Group_ppo": [df["nb_group_min"].min() for df in dfs_ppo],
                }
            )
            fig3 = get_box_modem_group_reward(
                dfs_best,
                "Boxplot Actor Critic",
                ["Modem_actor", "Group_actor", "Modem_ppo", "Group_ppo"],
            )

    return table, style_box, fig1, fig2, fig3


if __name__ == "__main__":
    app.run_server(debug=True)
