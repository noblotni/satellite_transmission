import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import os

def generate_report(csv_file_path, report_path, report_title, params):
    df = pd.read_csv(csv_file_path)
    df = df.reset_index(drop=True)

    df_episode_reward = df.groupby("episode")["reward"].max()
    df_episode_modem = df.groupby("episode")["nb_modem_min"].min()
    df_episode_group = df.groupby("episode")["nb_group_min"].min()
    
    graph1 = px.line(df, x=df.index, y="reward", title="Reward vs. Timestep")
    graph2 = px.line(df, x=df.index, y="nb_modem_min", title="nb_modem_min vs. Timestep")
    graph3 = px.line(df, x=df.index, y="nb_group_min", title="nb_group_min vs. Timestep")
    graph4 = px.line(df_episode_reward, x=df_episode_reward.index, y="reward", title="Reward vs. Episode")
    graph5 = px.line(df_episode_modem, x=df_episode_modem.index, y="nb_modem_min", title="nb_modem_min vs. Episode")
    graph6 = px.line(df_episode_group, x=df_episode_group.index, y="nb_group_min", title="nb_group_min vs. Episode")

    with open(report_path, "w") as report_file:
        report_file.write(f"<html><head><title>{report_title}</title></head><body>")
        report_file.write(f"<h1 align='center'>{report_title}</h1>")

        report_file.write('<table style="border-collapse: collapse; width: 35%; margin: 0 auto; text-align: center;">')
        report_file.write('<tr><th style="border: 2px solid black; padding: 8px; font-weight: bold;">Experimental design</th><th style="border: 2px solid black; padding: 8px; font-weight: bold;;">Value</th></tr></thead>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of links</td><td style="border: 1px solid black; padding: 8px;">{params["Number of links"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of modems</td><td style="border: 1px solid black; padding: 8px;">{params["Number of modems"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of groups</td><td style="border: 1px solid black; padding: 8px;">{params["Number of groups"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Time</td><td style="border: 1px solid black; padding: 8px;">{params["Time"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of episodes</td><td style="border: 1px solid black; padding: 8px;">{params["Number of episodes"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of timesteps</td><td style="border: 1px solid black; padding: 8px;">{params["Number of timesteps"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of runs</td><td style="border: 1px solid black; padding: 8px;">{params["Number of runs"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Time out</td><td style="border: 1px solid black; padding: 8px;">{params["Time out"]}</td></tr></table>')

        report_file.write('<h2 align="center">Results</h2>')
        report_file.write('<h3 align="center">Episode</h3>')
        report_file.write(graph4.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graph5.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graph6.to_html(full_html=False, include_plotlyjs="cdn"))

        report_file.write('<h3 align="center">Timestep</h3>')
        report_file.write(graph1.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graph2.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graph3.to_html(full_html=False, include_plotlyjs="cdn"))
        
        report_file.write("</body></html>")
        
    print(f"Report saved to {os.path.abspath(report_path)}")