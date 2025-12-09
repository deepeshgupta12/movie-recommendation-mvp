from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    """
    Convenience runner for Streamlit UI.
    """
    app_path = os.path.join("ui", "streamlit_app.py")
    if not os.path.exists(app_path):
        raise FileNotFoundError("ui/streamlit_app.py not found.")

    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()