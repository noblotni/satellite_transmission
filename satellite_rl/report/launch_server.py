import subprocess
import time
from pathlib import Path

import requests


def launch_server():
    """Launch the Dash server."""
    i = 0
    server_launched = False
    while not server_launched:
        try:
            response = requests.get("http://localhost:8050/")
            if response.status_code == 200:
                server_launched = True
        except:
            pass
        if not server_launched:
            report_path = str(Path.cwd() / "satellite_rl" / "report" / "report_dashboard.py")
            cmd = "python " + report_path
            _ = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(3)
            i += 1
            if i == 3:
                break
    return server_launched
