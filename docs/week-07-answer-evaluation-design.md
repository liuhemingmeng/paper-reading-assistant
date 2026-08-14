# 第 7 周设计：答案质量评测（Answer Quality Evaluation）

第 5 周解决了"检索是否命中目标页"（Recall@K / MRR），第 6 周让 embedding 可替换。但 RAG 系统的最终产物是**答案**，而答案可能：

1. **编造来源**：声称"见第 5 页"，但检索根本没返回第 5 页。
2. **幻觉事实**：答案包含证据之外的内容。

第 7 周在既有检索评测之上，补齐**生成质量**的两个可量化维度，形成端到端评测闭环。

## 目标

- 离线可验证 **Citation Correctness（引用正确性）**：答案提及的页码必须 ∈ 检索证据页码。
- 可选 **Faithfulness（忠实度）**：用 LLM 裁判判断答案是否仅基于证据。
- 输出单一 **AnswerEvaluationReport**：同时含检索指标与答案质量指标，便于在同一固定评测集上对比。

## 设计决策

### 1. Citation Correctness 完全离线

不依赖任何外部服务。流程：

```
生成答案文本 → 用正则抽取显式页码提及
  （支持 page N / p.N / p N / 第N页）
→ 与检索证据页码集合比对
→ 任何提及页 ∉ 证据页 ⇒ unsupported_pages，consistent = False
```

- 答案不提页码时 `cited_pages == []`，视为 `consistent=True`（无违反），但调用方可据此判断"答案未显式归因"。
- 这是对"可解释 RAG 不可凭空引用"的最基础、零成本检查。

### 2. Faithfulness 用 LLM-as-Judge，优雅降级

- 新增 `FaithfulnessJudge` 协议与 `OpenAICompatibleFaithfulnessJudge`（复用 `LLM_*` 配置与 httpx 重试模式）。
- 裁判输入：`(question, evidence_text, answer)`，要求模型返回 JSON `{faithful: bool, reason: str}`。
- `evaluate_answers(..., judge=None)`：未传 judge 时 `faithfulness_run=False`、`faithful_rate=None`，报告仍输出检索与引用指标。
- 未配置 LLM 时 API 返回 `503`，不误请求未知服务。

### 3. 复用而非重写

- `evaluate_answers` 内部调用第 5 周的 `evaluate_retrieval`，检索指标（Recall@K/MRR）直接复用。
- 答案生成复用 `answer_question`，citations 与 evidence 一一对应（第 4 周契约）。
- `OpenAICompatibleFaithfulnessJudge` 与 `OpenAICompatibleClient` 共享配置读取与重试策略，但分属不同模块以保持边界清晰。

## 接口契约

### 新增请求/响应模型

```
AnswerEvaluationCaseRequest {
  question: str
  expected_page_numbers: list[int]      # 检索目标页（沿用第 5 周）
  expected_answer_pages: list[int] = []  # 可选：答案应引用的页
}
AnswerEvaluationCaseResultRead {
  question, expected_page_numbers, answer,
  cited_pages, evidence_pages,
  citation_consistent: bool,
  faithful: bool | None,
  faithfulness_reason: str | None
}
AnswerEvaluationReportRead {
  k, case_count,
  recall_at_k, mean_reciprocal_rank,
  citation_correct_rate,
  faithfulness_run: bool,
  faithful_rate: float | None,
  results: list[AnswerEvaluationCaseResultRead]
}
```

### 新增路由

```
POST /papers/{paper_id}/answers:evaluate
  body: list[AnswerEvaluationCaseRequest]
  query: k=3, run_faithfulness=false
  → 401 不存在 / 404 无文档
  → 409 未建索引
  → 422 空问题 / 无关证据
  → 503 未配置 LLM（生成答案）或未配置裁判（run_faithfulness=true）
  → 502 embedding / LLM 调用失败
```

### CLI

`scripts/evaluate_answers.py`：对本地数据库运行端到端评测。`--fake` 使用离线假生成器与假裁判，无需任何 LLM 凭据即可演示引用正确性链路。

## 非目标

- 不做答案与"标准答案"的语义相似度（需要人工标注或更强模型，留作后续）。
- 不自动修正幻觉，只检测并报告。
- 真实 embedding 与真实 LLM 裁判的量化对比仍依赖用户配置凭据后运行。

## 测试范围

- `extract_cited_pages` 的中英文页码抽取与去重。
- `check_citation_correctness` 对编造页码的识别。
- `evaluate_answers` 用 Fake 生成器跑通检索+引用指标（无 LLM）。
- `evaluate_answers` 用 Fake 裁判设置 `faithful_rate`。
- 路由：未配置 LLM 返回 503；未建索引返回 409；非法用例返回 422。
- `OpenAICompatibleFaithfulnessJudge.from_environment()` 在缺失配置时抛 `LLMNotConfiguredForJudgingError`。
