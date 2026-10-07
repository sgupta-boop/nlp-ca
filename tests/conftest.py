import sys
from pathlib import Path

# make `from src import ...` work when pytest is run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
