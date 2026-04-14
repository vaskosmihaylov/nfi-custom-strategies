import sys
from pathlib import Path


AI_STRATEGY_DIR = Path(__file__).resolve().parents[1] / "Ai"
if str(AI_STRATEGY_DIR) not in sys.path:
    sys.path.append(str(AI_STRATEGY_DIR))

from fibbo import Fibbo as BaseFibbo


class Fibbo(BaseFibbo):
    pass


__all__ = ["Fibbo"]
