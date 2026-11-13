"""
start.py — One-click launcher for Academic Document Assistant.
Bypasses first-run onboarding email prompts and Windows UTF-8 console issues.
"""
import os
import sys
from pathlib import Path

# Force UTF-8 on Windows console
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Pre-activate Streamlit credentials to bypass email prompt
from streamlit.runtime.credentials import Credentials, _Activation

c = Credentials.get_current()
c.activation = _Activation(email="", is_valid=True)

from streamlit.web import cli as stcli

if __name__ == "__main__":
    app_path = str(Path(__file__).parent.resolve() / "app.py")
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())
