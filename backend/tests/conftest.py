import sys
from pathlib import Path

# Allow importing from backend/ directly (e.g. `from rag import ...`)
sys.path.insert(0, str(Path(__file__).parent.parent))
