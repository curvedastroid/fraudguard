"""Compatibility entrypoint for hosts that run `streamlit run app.py`.

The real application lives in streamlit/app.py so it can keep its page modules
and utilities together. This wrapper lets Render or local users start the app
from the repository root without changing the app's path assumptions.
"""

from pathlib import Path
import runpy


APP_PATH = Path(__file__).resolve().parent / "streamlit" / "app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
