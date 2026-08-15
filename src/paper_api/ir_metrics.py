"""检索基准用的信息检索评测指标（纯 Python，无第三方依赖）。

所有指标都接收「候选 id 的有序列表」和「相关 id 集合」，相关性视为二元
（相关 / 不相关）。nDCG 使用二元增益，便于在已知项检索（known-item
search）这一类任务上直接解释。
"""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """前 k 个候选里覆盖到的相关项比例（按相关项总数归一）。"""
    if not relevant:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for cand in top if cand in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """第一个相关项的倒数排名；前 k 个里没有相关项则为 0。"""
    for idx, cand in enumerate(ranked[:k], start=1):
        if cand in relevant:
            return 1.0 / idx
    return 0.0


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """归一化折损累计增益；理想排序由相关项数量决定。"""
    gains = [1.0 if cand in relevant else 0.0 for cand in ranked[:k]]
    dcg = _dcg(gains)
    ideal = _dcg([1.0] * min(len(relevant), k))
    return dcg / ideal if ideal > 0 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = ["mrr_at_k", "ndcg_at_k", "recall_at_k", "mean"]
