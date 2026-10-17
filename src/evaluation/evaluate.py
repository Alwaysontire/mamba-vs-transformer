"""Evaluation"""



from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.common.io import now_iso, write_json
from src.evaluation.metrics import compute_perplexity

