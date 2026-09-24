#!/usr/bin/env python3
"""
Simple Python wrapper for Google Elo C++ implementation.
"""

import subprocess
import tempfile
import os
import pandas as pd
import re
from pathlib import Path
import time
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import SCRATCH_TEMP_DIRECTORY

# Python 和 C++ Google ELO 之间的桥，本身不改算法，只负责格式转换、调用二进制、解析输出

# 它负责把 Python 数据喂给已有 Google ELO，然后把排名分数和置信区间拿回来

class GoogleEloWrapper:
    """Simple wrapper for Google Elo C++ implementation."""
    
    def __init__(self):
        """Initialize the wrapper."""
        script_dir = Path(__file__).parent
        self.elo_binary_path = script_dir / "google_elo" / "build" / "elo_main"
        self.last_rater_random_probabilities = {}
    
    def run_elo_analysis(
        self, df: pd.DataFrame, settings: dict = None
    ) -> pd.DataFrame:
        """Run Google Elo analysis on a DataFrame.
        
        Args:
            df: DataFrame in Google Elo format (methodA, methodB, answerValue, answerer)
            
        Returns:
            DataFrame with ELO scores and confidence intervals
        """
        
        # Create temporary CSV file
        temp_dir = SCRATCH_TEMP_DIRECTORY
        os.makedirs(temp_dir, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, dir=temp_dir) as f:

            # if isGolden is True, change it to true
            df['isGolden'] = df['isGolden'].astype(str).replace('True', 'true')
            # in first two columns if they are empty then insert N/A
            df['methodA'] = df['methodA'].fillna('N/A')
            df['methodB'] = df['methodB'].fillna('N/A')
            df.to_csv(f, index=False, header=True)
            temp_csv_path = f.name
        
        try:
            # Run the elo binary
            # change True to true in isGolden column
            df['isGolden'] = df['isGolden'].astype(str).replace('True', 'true')
            start = time.time()
            command = [
                str(self.elo_binary_path),
                f"--csv_path={temp_csv_path}",
            ]
            if settings:
                allowed = {"alpha", "rater_prior_strength"}
                unknown = set(settings) - allowed
                if unknown:
                    raise ValueError(
                        f"Unknown Google ELO settings: {sorted(unknown)}"
                    )
                for key in sorted(settings):
                    command.append(f"--{key}={float(settings[key])}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            end = time.time()
            delta_time = end - start

            # Keep the fitted rater parameters for held-out prediction while
            # preserving the existing ranking return type.
            self.last_rater_random_probabilities = (
                self._parse_rater_random_probabilities(result.stdout)
            )
            return self._parse_elo_output(result.stdout), delta_time
            
        except subprocess.CalledProcessError as e:
            print(f"Error running Google Elo: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return pd.DataFrame(), 0
        
        finally:
            # Each bootstrap creates a temporary CSV; remove it after C++ finishes.
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
    
    def _parse_elo_output(self, output: str) -> pd.DataFrame:
        """Parse the output from the Google Elo binary and return a DataFrame.
        
        Args:
            output: Raw output from the elo binary
            
        Returns:
            DataFrame with ELO scores
        """
        lines = output.strip().split('\n')
        
        elo_scores = []
        number = (
            r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)"
            r"(?:[eE][-+]?\d+)?|nan|inf)"
        )
        score_line = re.compile(
            rf"^(?P<method>.+):\s*"
            rf"(?P<score>{number})\s*"
            rf"\[\s*(?P<low>{number})(?:\s*,|\s+-)\s*"
            rf"(?P<high>{number})\s*\]\s*\Z",
            flags=re.IGNORECASE,
        )
        
        for line in lines:
            line = line.strip()

            # Stop at empty lines or section headers
            if len(line) == 0:
                break
                
            # Model identifiers may contain colons, for example
            # model::agent_type. Match the score payload from the end of
            # the line instead of splitting at the first colon.
            match = score_line.fullmatch(line)
            if match is None:
                raise ValueError(f"Unexpected Google ELO output line: {line!r}")
            elo_scores.append({
                'Method': match.group('method').strip(),
                'ELO Score': float(match.group('score')),
                'Lower CI (99%)': float(match.group('low')),
                'Upper CI (99%)': float(match.group('high'))
            })
        
        df = pd.DataFrame(elo_scores)
        return df.sort_values('ELO Score', ascending=False).reset_index(drop=True)


    def _parse_rater_random_probabilities(self, output: str) -> dict:
        """Parse the Crowd-BT per-rater random-answer probabilities."""
        marker = "Rater random probabilities"
        if marker not in output:
            return {}

        section = output.split(marker, 1)[1]
        if "Suggestions" in section:
            section = section.split("Suggestions", 1)[0]

        probabilities = {}
        for raw_line in section.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            rater, raw_value = line.rsplit(":", 1)
            try:
                probabilities[rater.strip()] = float(raw_value.strip())
            except ValueError:
                continue
        return probabilities

