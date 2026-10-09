"""Run the reflection scenario."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from furhat_interaction.app import run_demo

if __name__ == "__main__":
    raise SystemExit(run_demo(1, "reflection", sys.argv[1:]))
