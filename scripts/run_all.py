from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scaa_us_weather_yield.pipeline import run_all  # noqa: E402


if __name__ == "__main__":
    run_all(PROJECT_ROOT)
