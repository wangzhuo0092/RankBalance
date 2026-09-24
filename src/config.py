"""Portable paths used by optional native baseline wrappers."""

import os
from pathlib import Path


PROJECT_ROOT_DIRECTORY = str(Path(__file__).resolve().parents[1])
SCRATCH_TEMP_DIRECTORY = os.environ.get("RANKBALANCE_TMPDIR", "/tmp/rankbalance")
USER_BASHRC_PATH = os.environ.get("BASH_ENV", "")
