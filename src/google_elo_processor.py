#!/usr/bin/env python3
"""
Data processor for Google Elo format.

This module handles loading and processing data in the Google Elo format,
providing compatibility with the existing Bayesian Elo framework.
"""
# 封装一次 Google ELO 处理流程，把 DataFrame 交给 GoogleEloWrapper，拿回 ELO 结果。
# google_elo_processor.py 是 Google ELO 的处理层（调用 GoogleEloWrapper），但逻辑很轻，真正调用 C++ 的活在 google_elo_wrapper.py，真正算法在 src/google_elo/ 

"""
google_elo_processor.py：处理层 / 接口层
它负责接在项目主流程里，比如 bootstrap.py 里最终会调用它。
它做的事：
保存 df 和 valid_raters
检查数据是不是空
调用 GoogleEloWrapper
把结果返回给上层
它更像“项目内部统一接口”。

google_elo_wrapper.py：调用层 / 桥接层
它负责真正和 C++ 程序交互。
它做的事：
把 DataFrame 写成临时 CSV
调用 C++ 可执行文件 elo_main
读取 C++ stdout
解析成 DataFrame
返回 ELO 表和耗时
它更像“Python 调 C++ 的适配器”。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import tempfile
import os

class GoogleEloProcessor:
    """Main processor for running Google Elo analysis."""
    
    def __init__(self, df: pd.DataFrame, valid_raters: List[str]):
        """Initialize the processor.
        
        Args:
            df: DataFrame with comparison data
            valid_raters: List of valid rater IDs
        """
        self.df = df
        self.valid_raters = valid_raters
        self.rater_random_probabilities = {}
    
    def process(
        self,
        df: pd.DataFrame = None,
        valid_raters: List[str] = None,
        uncertainty: bool = False,
        settings: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, float]:
        """Process the data using Google Elo.
        
        Args:
            df: DataFrame with comparison data (optional, uses self.df if not provided)
            valid_raters: List of valid rater IDs (optional, uses self.valid_raters if not provided)
            model: Model name (should be 'google_elo')
            
        Returns:
            DataFrame with ELO scores and confidence intervals
        """
        if df is None:
            df = self.df
        if valid_raters is None:
            valid_raters = self.valid_raters
        
        if df.empty:
            print("No valid data found after filtering by raters")
            return pd.DataFrame(), 0
        
        try:
            # Run Google Elo analysis
            from google_elo_wrapper import GoogleEloWrapper
            wrapper = GoogleEloWrapper()
            results, time = wrapper.run_elo_analysis(
                df, settings=settings
            )
            self.rater_random_probabilities = (
                wrapper.last_rater_random_probabilities
            )
            
            # For Google ELO, we don't have built-in confidence intervals
            # but we can add estimated confidence intervals if uncertainty is requested
            if uncertainty and not results.empty:
                
                results['Lower CI (99%)'] = results['Lower CI (99%)'] 
                results['Upper CI (99%)'] = results['Upper CI (99%)']
            
            return results, time
            
        except Exception as e:
            print(f"Error running Google Elo analysis: {e}")
            return pd.DataFrame(), 0
