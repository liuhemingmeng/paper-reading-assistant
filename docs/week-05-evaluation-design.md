# 第 5 周设计：RAG 检索与引用评测

## 目标

将第 4 周“能检索、能回答”的能力转为“可以判断检索是否变好”的能力。评测输入是一组人为标注的 `question -> expected_page_numbers`，输出逐题命中明细和汇总指标：Recall@K、MRR。

本周首先评测 retrieval，因为回答生成依赖外部 LLM，且“答案正确”需要更复杂的人工或模型裁判。第 4 周已保证回答返回的 citations 与实际送入模型的 evidence 来自同一个列表；这构成当前引用一致性的代码级保障。

## 评测接口

```text
POST /papers/{paper_id}/retrieval:evaluate?k=3
Content-Type: application/json

[
  {
    "question": "What finds relevant evidence?",
    "expected_page_numbers": [1]
  }
]
```

输出：

```json
{
  "k": 3,
  "case_count": 1,
  "recall_at_k": 1.0,
  "mean_reciprocal_rank": 1.0,
  "results": [
    {
      "question": "What finds relevant evidence?",
      "expected_page_numbers": [1],
      "retrieved_page_numbers": [1, 2],
      "hit": true,
      "reciprocal_rank": 1.0
    }
  ]
}
```

## 标注单位：为什么先用页码

当前系统从 PDF 提取文本时已经稳定保存 `page_number`，因此第 5 周用页码作为 relevance label：易于人工在原 PDF 中核验，也不依赖 chunk 主键（重传文档后主键可能变化）。

页码比 chunk 级标注粗：同一页可能有多个 chunk，其中并非都能回答问题。因此它适合当前的最小评测闭环；后续可以升级为 `(document hash, page, chunk sequence)` 或原文字符范围标注。

## 指标

对于问题 i，R_i 是期望页码集合，检索前 K 个结果的页码序列为 L_i：

- `Hit@K(i) = 1`，当且仅当 L_i 中至少有一个页码属于 R_i。
- `Recall@K = mean(Hit@K(i))`。本项目每题至少标一个目标页，因此这里是“问题级 evidence recall”。
- `rank_i` 是第一个相关页码出现在 L_i 的位置；若不存在则为无穷大。
- `RR_i = 1 / rank_i`；没有命中时为 0。
- `MRR = mean(RR_i)`。

MRR 比 Recall@K 多反映了排序：同样全部命中时，目标证据排第 1 比排第 3 得分更高。

## 错误边界

| 状况 | 状态码 | 原因 |
| --- | --- | --- |
| 论文或文档不存在 | 404 | 没有评测对象 |
| 文档未建立索引 | 409 | 先调用 retrieval:index |
| 空评测集、空问题、非法页码、无相关证据 | 422 | 评测输入或检索结果不具备意义 |
| K 不在 1-10 | 422 | 与检索接口保持同一上下文上限 |

## 可复现性

- 当前 embedding 为本地确定性 hashing vector，同一文本和问题得到相同排序。
- 评测请求显式携带 cases，不把“正确答案”写死在业务代码或测试 fixture 中。
- 返回每题的 retrieved pages，使汇总数值可审计，而不是只给一个分数。
- 单元测试覆盖全命中、首次相关结果位于第 2 名、未命中、未建索引和非法输入。

## 已知限制

- 只评测 retrieval 的页码命中，不直接证明生成答案是否忠实、完整或表达良好。
- 当前系统出现无词面相关证据时返回 422；真实评测集也需要记录此类失败，而非静默把它当作普通空结果。
- 单篇、少量 chunks 时线性扫描足够；多文档规模需引入持久向量库和批量评测运行器。
- 当前 hashing baseline 不适合用绝对分数比较真实语义质量；它的价值是为后续真实 embedding 提供可对照的评测框架。

## 下一步

第 6 周将引入真实 embedding 的可替换实现，使用同一份固定评测集对比 Recall@K 和 MRR；再扩展 citation correctness 与 answer faithfulness 的人工标注或裁判模型评测。
