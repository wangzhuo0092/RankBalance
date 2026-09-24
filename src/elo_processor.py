from typing import Dict, List, Tuple, Optional, DefaultDict
import pandas as pd
import numpy as np
from bayesian_elo import update_bayesian_elo as update_bayesian_elo
from bayesian_elo_noise_vectorized import update_bayesian_elo as update_bayesian_elo_noise
from bayesian_elo_calibrated import (
    update_bayesian_elo as update_bayesian_elo_correctness,
    update_bayesian_elo_reverse as update_bayesian_elo_correctness_reverse,
    update_bayesian_elo_adaptive_clip,
    update_bayesian_elo_adaptive_flip,
)
from am_elo import update_am_elo, update_label_smoothed_bt, update_m_elo
from rank_balance import update_rank_balance
from google_elo_processor import GoogleEloProcessor
from models import (
    Metric, Session, Slate, Rating, Stimulus, QuerySet,
    MIN_COMPARISONS_PER_USER,
    elo_to_skill
)
from collections import defaultdict
import math
import os
import random
from multiprocessing import Pool
from tqdm import tqdm
from scipy.stats import spearmanr
import time
import argparse

"""
google_elo
= Crowd-BT baseline
= 点估计 + rater random probability

bayesian_elo
= Bayes-BT baseline
= Bayesian posterior，但不考虑 rater 噪声

bayesian_elo_noise
= BBQ
= Bayesian posterior + rater quality / noisy comparison model

bayesian_elo_correctness
= Correctness BBQ (downweight)
= rater label correctness eta + discard incorrect weight

bayesian_elo_correctness_reverse
= Correctness BBQ (reverse)
= rater label correctness eta + reverse incorrect weight

bayesian_elo_adaptive_clip
= Adaptive soft clipping
= estimate contamination ratio and downweight low clean-likelihood comparisons

bayesian_elo_adaptive_flip
= Adaptive soft flip correction
= estimate contamination ratio and reverse clipped weight

am_elo
= am-ELO baseline
= full-data maximum likelihood point estimate + per-rater discrimination

m_elo
= m-ELO baseline
= ordinary Bradley-Terry full-data maximum likelihood point estimate

label_smoothed_bt
= LS-BT baseline
= Bradley-Terry MLE with a fixed symmetric soft-label target
"""

# model_factory 是 Bayesian 系列模型的“名字 -> 函数”映射表。
# 后面 EloProcessor.process 会根据传入的 model 字符串，从这里取出真正要调用的算法函数。
model_factory = {
    'bayesian_elo': update_bayesian_elo,
    'bayesian_elo_noise': update_bayesian_elo_noise,
    'bayesian_elo_correctness': update_bayesian_elo_correctness,
    'bayesian_elo_correctness_reverse': update_bayesian_elo_correctness_reverse,
    'bayesian_elo_adaptive_clip': update_bayesian_elo_adaptive_clip,
    'bayesian_elo_adaptive_flip': update_bayesian_elo_adaptive_flip,
    'm_elo': update_m_elo,
    'am_elo': update_am_elo,
    'label_smoothed_bt': update_label_smoothed_bt,
    'rank_balance': update_rank_balance,
}
CORRECTNESS_MODELS = {
    'bayesian_elo_correctness',
    'bayesian_elo_correctness_reverse',
    'bayesian_elo_adaptive_clip',
    'bayesian_elo_adaptive_flip',
}
GPU_BAYESIAN_MODELS = CORRECTNESS_MODELS | {
    'bayesian_elo',
    'bayesian_elo_noise',
}

# elo_processor.py 的作用是：把原始 CSV 数据整理成各个 ELO 算法能吃的格式，然后根据 model 参数调用对应算法，最后把结果整理成表格
# 调用链是：
# bootstrap.py
#   -> EloProcessor.process(...)
#       -> model_factory[model](final_metric, final_sessions_qs)
#           -> 对应模型的 update_bayesian_elo(...)，例如 bayesian_elo_calibrated.update_bayesian_elo(...)


# DataProcessor 负责读取原始 CSV，并筛选出比较次数足够的有效用户。
# 它只做数据加载和基础校验，不负责真正的 ELO 算法。
class DataProcessor:
    """Handles data processing and validation."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        
    # load_data 负责读取 CSV，并检查必需字段是否存在。
    # 同时会调用 get_valid_users，把比较次数不足的用户过滤掉。
    def load_data(self) -> bool:
        """Load and validate the input data."""
        print(f"Loading data from {self.data_path}")
        if not os.path.exists(self.data_path):
            return False
            
        self.df = pd.read_csv(self.data_path)

        # All model paths use ``draw`` as the canonical tie label. In
        # particular, the Crowd-BT C++ parser otherwise interprets ``Tie`` as
        # an A win because it only recognizes the lowercase word ``draw``.
        if 'answerValue' in self.df.columns:
            normalized_answers = self.df['answerValue'].astype(str).str.strip()
            tie_mask = normalized_answers.str.lower().isin({'tie', 'draw'})
            self.df.loc[tie_mask, 'answerValue'] = 'draw'

        # 没有标注者 ID 时，把全部比较视为来自同一个全局标注者。
        if 'answerer' not in self.df.columns:
            self.df['answerer'] = 'global_rater'
        else:
            self.df['answerer'] = self.df['answerer'].astype(object)
            missing_answerer = (
                self.df['answerer'].isna()
                | self.df['answerer'].astype(str).str.strip().eq('')
            )
            self.df.loc[missing_answerer, 'answerer'] = 'global_rater'

        # drop users with less than MIN_COMPARISONS_PER_USER comparisons
        self.df = self.df[self.df['answerer'].isin(self.get_valid_users())]
        
        # Validate required columns for Google Elo format
        required_columns = ['methodA', 'methodB', 'answerValue', 'answerer']
        if not all(col in self.df.columns for col in required_columns):
            print(f"Missing required columns. Expected: {required_columns}")
            print(f"Found columns: {list(self.df.columns)}")
            return False
        
        return True
        
    # get_valid_users 统计每个 answerer 的比较次数。
    # 只有比较次数达到 MIN_COMPARISONS_PER_USER 的用户，才会参与后续 ELO 计算。
    def get_valid_users(self) -> List[str]:
        """Get list of users with sufficient non-training comparisons."""
        if self.df is None:
            return []
        
        # Count comparisons per rater
        user_comparisons = defaultdict(int)
        for _, row in self.df.iterrows():
            user_comparisons[row['answerer']] += 1
        
        return [user for user, count in user_comparisons.items() 
                if count >= MIN_COMPARISONS_PER_USER]
    
    

# EloWrapper 负责在“表格数据”和 Bayesian ELO 需要的内部对象之间做转换。
# 它会把 DataFrame 转成 Session / Slate / Rating 结构，也会把算法输出的 metric.state 转成结果表。
class EloWrapper:
    """Wrapper for ELO model operations."""

    # get_user_sessions 把 DataFrame 按评分者和比较题目分组。
    # 输出的是一个临时字典结构，后面会继续转成 Session / Slate / Rating 对象。
    def get_user_sessions(self, df: pd.DataFrame, valid_users: List[str]) -> Dict[str, List[Dict]]:
        """Get sessions for valid users in a data-oriented format, excluding training questions."""
        sessions = defaultdict(list)
        
        # Filter out training questions
        df_filtered = df[df['is_training'] == False]
        
        for session_id, session_group in df_filtered.groupby('session_id'):
            if session_id not in valid_users:
                continue
                
            for slate_id, slate_group in session_group.groupby('slate_id'):
                if len(slate_group) != 2:
                    continue
                    
                slate_data = {
                    'slate_id': slate_id,
                    'ratings': [
                        {
                            'score': float(row['score']),
                            # Preserve the full model identifier. Using basename
                            # merges repositories that share a model name and
                            # makes Bayesian outputs disagree with Crowd-BT.
                            'stimulus': str(row['stimulus']).strip() if pd.notna(row['stimulus']) else 'unknown'
                        }
                        for _, row in slate_group.iterrows()
                    ]
                }
                sessions[session_id].append(slate_data)
                
        return dict(sessions)
    
    # convert_to_elo_format 把表格数据转换成 Bayesian ELO 算法需要的对象格式。
    # 每个 rater 变成一个 Session，每次 pairwise comparison 变成一个 Slate。
    def convert_to_elo_format(self, df: pd.DataFrame, valid_users: List[str]) -> List[Session]:
        """Convert data-oriented format to ELO model format."""
        elo_sessions = []
        sessions_dict = self.get_user_sessions(df, valid_users)
        
        for rater_id, slates in sessions_dict.items():
            elo_slates = []
            for slate in slates:
                ratings = [
                    Rating(r['score'], Stimulus(r['stimulus']))
                    for r in slate['ratings']
                ]
                elo_slates.append(Slate(ratings))
            
            session = Session(elo_slates, rater=rater_id)
            session.id = rater_id
            elo_sessions.append(session)
            
        return elo_sessions
        
    @staticmethod
    # get_elo_model_df 把算法写在 metric.state['scores'] 里的结果整理成 DataFrame。
    # bootstrap.py 后面会继续用这个表计算排名稳定性和保存结果。
    def get_elo_model_df(metric: Metric) -> pd.DataFrame:
        """Return a DataFrame with the ELO model statistics."""
        if not hasattr(metric, 'state') or 'scores' not in metric.state:
            return pd.DataFrame(columns=['Method', 'ELO Score', 'Lower CI (99%)', 'Upper CI (99%)'])
            
        data = []
        for method, score_data in sorted(metric.state['scores'].items(), 
                                       key=lambda x: x[1]['value'], 
                                       reverse=True):
            data.append({
                'Method': method,
                'ELO Score': score_data['value'],
                'Lower CI (99%)': score_data['p005'],
                'Upper CI (99%)': score_data['p995']
            })
        return pd.DataFrame(data)


# EloProcessor 是统一调度入口：根据 model 参数选择具体算法。
# google_elo 直接交给 GoogleEloProcessor；其他方法会先转换数据，再调用对应的估计函数。
class EloProcessor:
    def __init__(self, df: pd.DataFrame, valid_users: List[str]):
        """Initialize the analyzer with data and valid users.
        
        Args:
            df: DataFrame containing the comparison data
            valid_users: List of valid user IDs
        """
        self.df = df
        self.valid_users = valid_users
        self.metric = None  # Store the metric for later access

    # process 是整个 ELO 计算的统一入口。
    # 它根据 model 参数决定调用 Google ELO，还是调用 Bayes-BT / BBQ / Correctness BBQ 方法。
    def process(
        self,
        df: pd.DataFrame = None,
        valid_users: List[str] = None,
        model: str = 'google_elo',
        uncertainty: bool = False,
        shared_rater: bool = False,
        device: str = 'auto',
        optimization_trace_config: Optional[Dict] = None,
        google_elo_settings: Optional[Dict[str, float]] = None,
        prior_config: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.10,
        rank_balance_config: Optional[Dict[str, float]] = None,
    ):
        if model == 'google_elo':
            # Use Google Elo processor directly
            google_processor = GoogleEloProcessor(df, valid_users)
            results, time = google_processor.process(
                df,
                valid_users,
                uncertainty=uncertainty,
                settings=google_elo_settings,
            )
            final_metric = Metric()
            final_metric.state['computation_backend'] = 'cpu'
            final_metric.state['rater_random_probabilities'] = (
                google_processor.rater_random_probabilities
            )
            final_metric.state['qualities'] = {
                rater: {'value': 1.0 - random_probability}
                for rater, random_probability
                in google_processor.rater_random_probabilities.items()
            }
            final_metric.state['quality_parameter'] = 'nonrandom_probability'
            final_metric.state['google_elo_settings'] = dict(
                google_elo_settings or {}
            )
            self.metric = final_metric
        else:
            # Convert to CLIC2024 format for Bayesian models
            clic2024_df = self._convert_google_elo_to_clic2024(df)
            elo_wrapper = EloWrapper()
            elo_sessions = elo_wrapper.convert_to_elo_format(clic2024_df, valid_users)
            final_metric = Metric()
            if optimization_trace_config:
                final_metric.state['_optimization_trace_config'] = dict(
                    optimization_trace_config
                )
            final_sessions_qs = QuerySet(elo_sessions)
            if model in CORRECTNESS_MODELS:
                time = model_factory[model](
                    final_metric,
                    final_sessions_qs,
                    shared_rater=shared_rater,
                    device=device,
                    prior_config=prior_config,
                )
            elif model in GPU_BAYESIAN_MODELS:
                time = model_factory[model](
                    final_metric,
                    final_sessions_qs,
                    device=device,
                    prior_config=prior_config,
                )
            elif model == 'label_smoothed_bt':
                time = model_factory[model](
                    final_metric,
                    final_sessions_qs,
                    label_smoothing=label_smoothing,
                )
                final_metric.state['computation_backend'] = 'cpu'
            elif model == 'rank_balance':
                time = model_factory[model](
                    final_metric,
                    final_sessions_qs,
                    config=rank_balance_config,
                )
            else:
                time = model_factory[model](final_metric, final_sessions_qs)
                final_metric.state['computation_backend'] = 'cpu'
            results = EloWrapper.get_elo_model_df(final_metric)
            if results.empty:
                return pd.DataFrame()
            if not uncertainty:
                results = results.drop(columns=['Lower CI (99%)', 'Upper CI (99%)'])
            self.metric = final_metric
        return results, time
    
    # _convert_google_elo_to_clic2024 把项目原始 CSV 格式转换成 Bayesian 模型使用的长表格式。
    # 一次 A/B 比较会被展开成两行：methodA 一行，methodB 一行。
    def _convert_google_elo_to_clic2024(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Google Elo format to CLIC2024 format."""
        if "isGolden" in df.columns:
            is_golden = (
                df["isGolden"].notna()
                & df["isGolden"].astype(str).str.strip().str.lower().eq("true")
            )
        else:
            is_golden = pd.Series(False, index=df.index)

        comparisons = df.loc[~is_golden]
        answers = comparisons["answerValue"].to_numpy()
        score_a = np.select([answers == "A", answers == "B"], [1.0, -1.0], default=0.0)
        score_b = -score_a
        session_ids = comparisons["answerer"].to_numpy()
        slate_ids = comparisons.index.to_numpy()

        row_a = pd.DataFrame({
            "score": score_a,
            "job_id": "nan",
            "session_id": session_ids,
            "rater_id": session_ids,
            "slate_id": slate_ids,
            "stimulus_id": comparisons["methodA"].to_numpy(),
            "stimulus": comparisons["methodA"].to_numpy(),
            "is_training": False,
        })
        row_b = pd.DataFrame({
            "score": score_b,
            "job_id": "nan",
            "session_id": session_ids,
            "rater_id": session_ids,
            "slate_id": slate_ids,
            "stimulus_id": comparisons["methodB"].to_numpy(),
            "stimulus": comparisons["methodB"].to_numpy(),
            "is_training": False,
        })
        return pd.concat([row_a, row_b], ignore_index=True)

    
