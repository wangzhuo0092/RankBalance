# 根据一批 noisy pairwise comparisons，同时估计 method 的 ELO 分数和 rater 的可靠程度。
import math
import logging
from functools import partial
import scipy.special
import scipy.stats
import numpy as np
from models import (
    Metric, QuerySet, ELO_INITIAL_SCORE, ELO_SCALE_FACTOR,
    ELO_MAX_UPDATES, ELO_CONVERGENCE_THRESHOLD,
    GAMMA_PRIOR_SHAPE, GAMMA_PRIOR_RATE,
    skill_to_elo, elo_to_skill, BETA_PRIOR_ALPHA, BETA_PRIOR_BETA,
    resolve_prior_config,
    initialize_optimization_state, optimization_trace_enabled,
    record_optimization_trace,
)
from torch_bayesian_backend import fit_original_bbq_cuda, resolve_device
# Metric, QuerySet：项目内部数据结构
# ELO_INITIAL_SCORE：初始 ELO，比如 2000
# ELO_SCALE_FACTOR：ELO 缩放系数，比如 400
# ELO_MAX_UPDATES：最大迭代次数
# ELO_CONVERGENCE_THRESHOLD：收敛阈值
# GAMMA_PRIOR_SHAPE, GAMMA_PRIOR_RATE：method skill 的 Gamma 先验参数
# BETA_PRIOR_ALPHA, BETA_PRIOR_BETA：rater quality 的 Beta 先验参数
# skill_to_elo, elo_to_skill：skill 和 ELO 的互转函数

from copy import deepcopy
import time

logger = logging.getLogger(__name__)

def update_bayesian_elo(metric, sessions, device='auto', prior_config=None):
    """Implements an iterative procedure for computing Elo scores.

    This algorithm is based on Caron & Doucet's (2010) Bayesian interpretation of an algorithm by
    Hunter (2004). However, where Caron & Doucet view it as expectation maximization algorithm for
    computing a MAP estimate, we here go a step further and interpret it as mean-field variational
    inference steps. This allows us to additionally compute uncercainty estimates.

    Args:
        metric: A Metric object containing the state
            metric 是一个结果容器。函数不会直接返回 scores，而是把 method 分数、
            rater quality、wins 矩阵等结果写进 metric.state。
        sessions: A QuerySet object containing the sessions to process
            sessions 是已经转换好的 pairwise comparison 数据；每个 session
            通常对应一个 rater，每个 slate 对应一次 A/B 比较。

    References:
        Caron & Doucet (2010), Efficient Bayesian Inference for the Bradley-Terry Model
        https://www.stats.ox.ac.uk/~doucet/caron_doucet_bayesianbradleyterry.pdf
    """

    # Parameters of a Gamma prior over skill values. The parameter "b" only determines the scale of
    # skill values. It should be of some numerical interest but should otherwise have no influence
    # on resulting Elo scores.
    start = time.time()

    priors = resolve_prior_config(prior_config)
    a, b = priors['gamma_shape'], priors['gamma_rate']
    alpha, beta = priors['beta_alpha'], priors['beta_beta']
    metric.state['prior_config'] = priors

    # scores 保存 method 的当前 ELO 估计，形如:
    # {'method_name': {'value': elo_score, 'p005': ..., 'p995': ...}}
    scores = metric.state.get('scores', {})
    # qualities 保存 rater 的当前可靠度估计 q_r，范围大致在 [0, 1]。
    # q_r 越高，表示这个 rater 的选择越像 Bradley-Terry 认真比较；
    # q_r 越低，表示越接近随机选择。
    qualities = metric.state.get('qualities', {})

    # methods / method_to_index 把字符串 method 名映射到矩阵下标。
    # 例如 methods = ['A', 'B'] 时，wins[0][1][r] 表示 rater r 认为 A 赢 B。
    methods = metric.state.get('methods', [])
    method_to_index = dict(zip(methods, range(len(methods))))

    # raters / rater_to_index 把评分者 ID 映射到 wins 的第三维下标。
    raters = metric.state.get('raters', [])
    rater_to_index = dict(zip(raters, range(len(raters))))

    # 三维胜负矩阵，最终形状为 (num_methods, num_methods, num_raters)。
    # wins[i][j][r] = rater r 判断 method i 赢 method j 的次数。
    wins = np.zeros((0, 0, 0))
    # Makes it so that the mode of the gamma distribution corresponds to an Elo score of
    # ELO_INITIAL_SCORE.
    # ELO_SCALE_FACTOR 控制 skill 差异被放大成多少 ELO 分；
    # 这段代码则用 offset 把 Gamma 先验的默认 skill 对齐到初始 ELO 2000。
    elo_offset = math.log10((a - 1) / b) * ELO_SCALE_FACTOR
    elo_offset = ELO_INITIAL_SCORE - elo_offset

    # 内部更新使用正数 skill，最后展示使用 ELO。
    # partial 这里固定 offset，后面可以直接调用 s2e(skill) / e2s(elo)。
    # 等价于先定义：
    # def s2e(skill):
    #     return skill_to_elo(skill, offset=elo_offset)
    s2e = partial(skill_to_elo, offset=elo_offset)
    e2s = partial(elo_to_skill, offset=elo_offset)

    # 把 sessions 里的对象结构转换成 wins 三维矩阵。
    # sessions
    # └── session：一个评分者的一批评价
    #         └── slate：一次 pairwise comparison
    #             └── ratings：这次比较里的两个候选对象

    for session in sessions:
        for slate in session.slates.all():
            ratings = slate.ratings.all()
            count = len(ratings)

            # BBQ 处理的是 pairwise comparison，所以一次 slate 必须刚好有两个候选项。
            if count != 2:
                logger.error(
                    'Slate {} in session {} has {} ratings instead of 2.'.format(
                        slate.id, session.id, count
                    )
                )
                continue
            
            # 谁评价的：rater
            # 比较对象 1：method_i
            # 比较对象 2：method_j
            # 对象 1 得分：s_i
            # 对象 2 得分：s_j
            rater = session.rater
            method_i = ratings[0].stimulus.name
            method_j = ratings[1].stimulus.name
            s_i, s_j = ratings[0].score, ratings[1].score

            # 第一次遇到某个 rater 时，把它加入 rater 列表，并扩展 wins 第三维。
            if rater not in rater_to_index:
                index_rater = len(raters)
                raters.append(rater)
                rater_to_index[rater] = index_rater
                # Expand 3rd dim
                wins = np.pad(wins, ((0, 0), (0, 0), (0, 1)))
            else:
                index_rater = rater_to_index[rater]

            # 第一次遇到某个 method 时，把它加入 method 列表。
            # wins 的前两维都表示 method，因此需要同时扩展行和列。
            if method_i not in method_to_index:
                index_i = len(methods)
                methods.append(method_i)
                method_to_index[method_i] = index_i
                wins = np.pad(wins, ((0, 1), (0, 0), (0, 0))) #新方法可能作为赢家，所以要给第 0 维加一格。
                wins = np.pad(wins, ((0, 0), (0, 1), (0, 0))) #新方法也可能作为输家，所以第 1 维也要加一格。 # Also expand dim 1 for symmetry
            else:
                index_i = method_to_index[method_i]

            # method_j 同理，确保两个被比较对象都在 wins 矩阵里有下标。
            if method_j not in method_to_index:
                index_j = len(methods)
                methods.append(method_j)
                method_to_index[method_j] = index_j
                wins = np.pad(wins, ((0, 1), (0, 0), (0, 0)))
                wins = np.pad(wins, ((0, 0), (0, 1), (0, 0)))  # Expand symmetrically
            else:
                index_j = method_to_index[method_j]

            s_i = ratings[0].score
            s_j = ratings[1].score

            # 把一次比较结果写入 wins:
            # score 高的一方记为 winner；平局时双方各加 0.5。
            if s_i > s_j:
                wins[index_i][index_j][index_rater] += 1
            elif s_j > s_i:
                wins[index_j][index_i][index_rater] += 1
            else:
                # We don't model ties explicitly. Instead, we consider the expected outcome of a
                # forced choice.
                wins[index_i][index_j][index_rater] += 0.5
                wins[index_j][index_i][index_rater] += 0.5

    #sort methods by name
    metric.state['methods'] = methods
    # wins = wins/100
    # wins = wins/wins.sum()
    
    # wins 的含义还是：wins[i][j][r] = rater r 判断 method i 赢 method j 的次数
    metric.state['wins'] = wins.tolist()

    n = len(methods)
    R = len(raters)
    initialize_optimization_state(
        metric,
        methods,
        scores,
        raters=raters,
        qualities=qualities,
        default_quality=0.5,
    )
    initial_elo = np.array([
        scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods
    ])
    record_optimization_trace(metric, 0, methods, initial_elo, wins)
    requested_device = 'cpu' if optimization_trace_enabled(metric) else device
    actual_device = resolve_device(requested_device)
    metric.state['computation_backend'] = actual_device

    if actual_device.startswith('cuda'):
        initial_elo = np.array([
            scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods
        ])
        initial_quality = np.array([
            qualities.get(r, {}).get('value', 0.5) for r in raters
        ])
        cuda_result = fit_original_bbq_cuda(
            wins,
            e2s(initial_elo),
            initial_quality,
            a=a,
            b=b,
            alpha=alpha,
            beta=beta,
            max_updates=ELO_MAX_UPDATES * sessions.count(),
            convergence_threshold=ELO_CONVERGENCE_THRESHOLD,
            elo_scale_factor=ELO_SCALE_FACTOR,
            device=actual_device,
        )
        for i, method in enumerate(methods):
            scores[method] = {'value': float(s2e(cuda_result['skill'][i]))}
        for r, rater in enumerate(raters):
            qualities[rater] = {'value': float(cuda_result['quality'][r])}

        for i, method_i in enumerate(methods):
            gamma_dist = scipy.stats.gamma(
                a=cuda_result['posterior_shape'][i],
                scale=1 / cuda_result['posterior_rate'][i],
            )
            percentiles = gamma_dist.ppf(
                [0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995]
            )
            # Preserve the original BBQ field mapping for reproducibility.
            scores[method_i]['p005'] = s2e(percentiles[0])
            scores[method_i]['p025'] = s2e(percentiles[0])
            scores[method_i]['p05'] = s2e(percentiles[1])
            scores[method_i]['median'] = s2e(percentiles[2])
            scores[method_i]['p95'] = s2e(percentiles[3])
            scores[method_i]['p975'] = s2e(percentiles[4])
            scores[method_i]['p995'] = s2e(percentiles[5])

        metric.state['scores'] = scores
        metric.state['qualities'] = qualities
        metric.state['rater_qualities'] = {
            r: qualities[r]['value'] for r in raters
        }
        metric.state['quality_parameter'] = 'nonrandom_probability'
        metric.state['iterations'] = cuda_result['iterations']
        metric.state['converged'] = cuda_result['converged']
        return time.time() - start

    # gamma[i][j][r] 是隐变量/责任权重:
    # 当 rater r 说 i 赢 j 时，gamma 表示该判断来自“认真比较”，而不是“随机乱选”的后验概率。
    gamma = np.zeros((n, n, R))

    # 核心训练/更新排名流程从这里开始：
    # 1. 读取当前 method ELO 和 rater quality。
    # 2. 根据当前状态计算 gamma，判断每条评价更像“认真比较”还是“随机选择”。
    # 3. 用 gamma 加权后的有效胜负，更新每个 method 的 skill / ELO。
    # 4. 再根据当前排名反过来更新每个 rater 的 quality。
    # 5. 如果本轮 ELO 变化足够小，就提前停止。
    # 这个循环会反复执行上述步骤，直到收敛或达到最大迭代次数。
    # 迭代更新 method skill 和 rater quality，直到 ELO 变化很小或达到最大迭代次数。
    converged = False
    for t in range(ELO_MAX_UPDATES * sessions.count()):
        max_elo_change = 0

        # 当前 ELO 分数；第一次运行时所有 method 使用 ELO_INITIAL_SCORE。
        elo_scores_t = np.array([scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods])
        # 转成 skill 空间，Bradley-Terry 概率用 skill_i / (skill_i + skill_j) 计算。
        skill_scores_t = e2s(elo_scores_t)
        # 当前 rater quality；第一次运行时默认 0.5，表示介于认真和随机之间。
        qualities_t = np.array([qualities.get(r, {}).get('value', 0.5) for r in raters])

        # Vectorized gamma computation
        # Create meshgrid for all i,j pairs
        i_indices, j_indices = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        # skill_ratios[i][j] = BT(i beats j) = skill_i / (skill_i + skill_j)。
        skill_ratios = skill_scores_t[i_indices] / (skill_scores_t[i_indices] + skill_scores_t[j_indices])
        # 扩展成 (1, 1, R)，方便和所有 i,j 组合广播计算。
        qualities_expanded = qualities_t[np.newaxis, np.newaxis, :]
        # 认真比较路径解释该回答的概率: q_r * BT(i beats j)。
        bt_win_prob = qualities_expanded * skill_ratios[..., np.newaxis]
        # 随机路径解释该回答的概率: (1 - q_r) * 0.5。
        # gamma 是“认真路径”在两种解释中的责任占比。
        gamma = bt_win_prob / (bt_win_prob + (1 - qualities_expanded) / 2)
        # method 不和自己比较，对角线没有意义，置零。
        for r in range(R):
            np.fill_diagonal(gamma[:, :, r], 0)

        # Vectorized score updates
        # 根据 BBQ / Bayesian BT 的更新式，更新每个 method 的 Gamma 后验参数。
        # nominator 近似是先验胜场 + 被 gamma 加权后的有效胜场。
        nominator = a - 1 + np.sum(wins * gamma, axis=(1, 2))
        # denominator 汇总该 method 与所有对手的有效比较量，用于得到新的 skill。
        denominator = b + np.sum(
            (wins * gamma + np.transpose(wins, (1, 0, 2)) * np.transpose(gamma, (1, 0, 2))) / 
            (skill_scores_t[:, np.newaxis, np.newaxis] + skill_scores_t[np.newaxis, :, np.newaxis]),
            axis=(1, 2)
        )
        
        # 新 skill = 后验参数的比值；再转回 ELO 尺度。
        skill_scores_new = nominator / denominator
        elo_scores_new = s2e(skill_scores_new)
        
        # Update max_elo_change
        max_elo_change = np.max(np.abs(elo_scores_new - elo_scores_t))
        
        # 写回 scores 字典，后续上层会从 metric.state['scores'] 取结果表。
        for i, method in enumerate(methods):
            scores[method] = {'value': elo_scores_new[i]}
        
        # Vectorized quality updates
        # 只取 i < j，避免同一对 method 的正反方向被重复计数。
        i_less_j = np.triu_indices(n, k=1)
        # rater quality 的 Beta 后验更新:
        # nominator_q 是先验成功次数 + 该 rater 的有效认真判断次数。
        nominator_q = alpha - 1 + np.sum(
            (wins[i_less_j] * gamma[i_less_j] + 
             wins[i_less_j[1], i_less_j[0]] * gamma[i_less_j[1], i_less_j[0]]),
            axis=0
        )
        # denominator_q 是先验总量 + 该 rater 的全部比较次数。
        denominator_q = alpha + beta - 2 + np.sum(
            wins[i_less_j] + wins[i_less_j[1], i_less_j[0]],
            axis=0
        )
        
        # 新 quality q_r = Beta 后验均值/众数形式的更新结果。
        qualities_new = nominator_q / denominator_q
        for r, rater in enumerate(raters):
            qualities[rater] = {"value": qualities_new[r]}

        # 如果这一轮所有 method 的 ELO 最大变化已经很小，就认为收敛。
        iteration_converged = max_elo_change < ELO_CONVERGENCE_THRESHOLD
        current_elo = np.array([scores[m]['value'] for m in methods])
        record_optimization_trace(
            metric, t + 1, methods, current_elo, wins,
            max_elo_change=max_elo_change, converged=iteration_converged,
        )
        if iteration_converged:
            converged = True
            break


    end = time.time()
    delta_time = end - start
    
    # 把最终 rater quality 单独存一份，便于结果分析。
    metric.state['rater_qualities'] = {r: qualities.get(r, {}).get('value', np.array(0.5)).item() for r in raters}
    metric.state['quality_parameter'] = 'nonrandom_probability'
    
    # Estimate uncertainty.
    # 使用最终的 skill / quality 再算一次 gamma，并用 Gamma 分布近似每个 method 的后验不确定性。
    for i in range(n):
    # 这段是在算法收敛后，为每个 method 估计不确定性/置信区间。
    # 这段不是继续训练/更新排名，而是在最终排名基础上，用 Gamma 后验分布估计每个 method 的不确定性，并把分位数保存到结果里。
        method_i = methods[i]
        # skill_i = e2s(scores.get(method_i, {}).get('value', 1))
        # a_i = a - 1 + np.sum(wins * gamma, axis=(1, 2))[i].item()

        # Evi = np.sum(wins * gamma, axis=(1, 2))[i].item()/100
        # Ezi = np.sum(
        #     (wins * gamma + np.transpose(wins, (1, 0, 2)) * np.transpose(gamma, (1, 0, 2))) / 
        #     (skill_scores_t[:, np.newaxis, np.newaxis] + skill_scores_new[np.newaxis, :, np.newaxis]),
        #     axis=(1, 2)
        # )[i].item()*100
        # print(gamma[0])
        # print((wins * gamma + np.transpose(wins, (1, 0, 2)) * np.transpose(gamma, (1, 0, 2)))[0])
        # print((wins+np.transpose(wins, (1, 0, 2)))[0])
        # asd
        # print(Evi, 1/(b+Ezi), Ezi)

        # a_i = a - 1 + wins.sum(axis=(1, 2))[i]
        # print(a_i, skill_scores_t[i]/a_i)

        # 重新拿最终的：method ELO和rater quality。并把 ELO 转回 skill，因为后验计算在 skill 空间里做
        elo_scores_t = np.array([scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods])
        skill_scores_t = e2s(elo_scores_t)
        qualities_t = np.array([qualities.get(r, {}).get('value', 0.5) for r in raters])

        # Vectorized gamma computation
        # Create meshgrid for all i,j pairs
        i_indices, j_indices = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        # 与上面的迭代部分相同：重新计算最终状态下的 BT 概率和 gamma。
        skill_scores_t = e2s(elo_scores_t)
        # 重新计算 Bradley-Terry 概率：
        skill_ratios = skill_scores_t[i_indices] / (skill_scores_t[i_indices] + skill_scores_t[j_indices])
        # Expand qualities to match dimensions
        # 重新计算最终状态下的 gamma。也就是：每个 rater 的每个胜负判断，有多大概率来自认真比较，而不是随机选择。
        qualities_expanded = qualities_t[np.newaxis, np.newaxis, :]
        # Compute bt_win_prob for all i,j,r
        bt_win_prob = qualities_expanded * skill_ratios[..., np.newaxis]
        # Compute gamma values
        gamma = bt_win_prob / (bt_win_prob + (1 - qualities_expanded) / 2)
        # Set diagonal to 0 for each rater
        for r in range(R):
            np.fill_diagonal(gamma[:, :, r], 0)

        # 计算每个 method 的 Gamma 后验参数。含义大概是：先验胜场 + gamma 加权后的有效胜场
        nominator = a - 1 + np.sum(wins * gamma, axis=(1, 2))
        # denominator 汇总该 method 与所有对手的有效比较量，用于得到新的 skill。
        # 这里是每个 method 的 Gamma 后验 rate 参数。含义大概是：先验 rate + 和所有对手比较带来的归一化项
        denominator = b + np.sum(
            (wins * gamma + np.transpose(wins, (1, 0, 2)) * np.transpose(gamma, (1, 0, 2))) / 
            (skill_scores_t[:, np.newaxis, np.newaxis] + skill_scores_t[np.newaxis, :, np.newaxis]),
            axis=(1, 2)
        )

        # Gamma distribution approximating the posterior distribution over the score.
        # 这里 nominator[i] 是 shape，denominator[i] 是 rate，所以 scale = 1 / rate。
        
        gamma_dist = scipy.stats.gamma(a=nominator[i], scale=1/denominator[i])
        # 取多个分位数，用来给 ELO 输出置信区间/中位数等统计量。
        percentiles = gamma_dist.ppf([0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995])

        # Convert percentiles to Elo scale.
        scores[method_i]['p005'] = s2e(percentiles[0])
        scores[method_i]['p025'] = s2e(percentiles[0])
        scores[method_i]['p05'] = s2e(percentiles[1])
        scores[method_i]['median'] = s2e(percentiles[2])
        scores[method_i]['p95'] = s2e(percentiles[3])
        scores[method_i]['p975'] = s2e(percentiles[4])
        scores[method_i]['p995'] = s2e(percentiles[5])

    metric.state['scores'] = scores
    metric.state['qualities'] = qualities
    metric.state['iterations'] = t + 1 if n else 0
    metric.state['converged'] = converged
    return delta_time
