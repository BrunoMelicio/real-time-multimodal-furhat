"""Run a preserved conference scene through the common entrypoint."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from furhat_interaction.app import main
if __name__=="__main__":
    raise SystemExit(main(["--people",str(2),"--part",str(1),*sys.argv[1:]]))
