# 第 6 周设计：可替换的真实 Embedding 客户端

目标：在不改变第 4 周检索业务与第 5 周评测集的前提下，把“本地 hashing 基线”替换为“OpenAI 兼容的真实 embedding 模型”，从而能用同一套 `Recall@K` / `MRR` 对比两种索引的质量。

## 设计原则

1. **业务只依赖协议，不依赖模型**
   - `retrieval.py` 中的 `TextEmbedder` 协议只有 `embed(text) -> EmbeddedText` 一个方法。
   - 服务层 `build_retrieval_index`、`retrieve_chunks`、`answer_question` 已经接受 `embedder` 参数，默认是 `LocalHashingEmbedder()`。
   - 第 6 周只新增“另一种实现”，不修改检索算法本身。

2. **配置驱动的选择**
   - 新增 `embeddings.py`：
     - `OpenAICompatibleEmbedder`：实现 `TextEmbedder`，调用 `POST {EMBEDDING_BASE_URL}/embeddings`。
     - `get_default_embedder()`：当 `EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY`、`EMBEDDING_MODEL` 三者都配置时返回真实客户端；否则回退到 `LocalHashingEmbedder()`。
   - `create_app(embedder=...)` 在启动时确定一次 `app.state.embedder`，整生命周期内索引与查询共用同一个嵌入器。

3. **索引与查询必须同源**
   - 同一运行实例中，索引向量和查询向量由同一个嵌入器产生，余弦相似度才有意义。
   - `retrieve_chunks` 增加一致性保护：若查询所用嵌入器的 `model` 与已建索引记录的 `model` 不同，直接返回 `409`，提示“重建索引”，避免维度不匹配导致无意义的相似度分数。

## 接口契约

| 环境变量 | 作用 | 缺省 |
| --- | --- | --- |
| `EMBEDDING_BASE_URL` | 兼容 OpenAI 的 embeddings 端点前缀 | 未设置 → 本地基线 |
| `EMBEDDING_API_KEY` | Bearer Token | 未设置 → 本地基线 |
| `EMBEDDING_MODEL` | 模型名，也作为向量来源标识写入 `ChunkEmbedding.model` | 未设置 → 本地基线 |

真实 embedding 失败（超时、网络错误、429、5xx）最多重试 3 次并退避；最终失败抛出 `EmbeddingResponseError`，索引与检索路由返回 `502`。这与第 3 周 LLM 客户端的错误处理保持一致。

## 非目标

- 不内置向量数据库：本项目的向量仍存于 SQLite 的 `ChunkEmbedding.vector_json`，重点是走通“可替换嵌入器 + 可回归评测”的端到端链路。
- 不自动做语义分块：分块策略沿用第 3 周，embedding 只改变“每段文本如何变成向量”。
- 不引入新的第三方依赖：复用已有的 `httpx`。

## 评测对比方式

配置真实 embedding 后：

1. 重新对论文调用 `POST /papers/{id}/retrieval:index`（此时索引会用真实模型重建）。
2. 用第 5 周同一份固定问题集调用 `POST /papers/{id}/retrieval:evaluate?k=...`。
3. 对比 `Recall@K`、`MRR` 与本地 baseline 的差异：真实语义向量通常在同义、改写、跨语言表述上明显优于词面 hashing。

> 注意：仓库未配置真实 embedding 凭据，因此“对比数值”需要你在本地填写 `.env` 后运行；代码已保证切换路径可用且可测试。

## 测试范围

- 单元测试：`OpenAICompatibleEmbedder.embed` 返回归一化向量；`get_default_embedder` 在有无配置下分别返回真实/本地嵌入器。
- 集成测试：注入 `LabeledFakeEmbedder` 证明 `TextEmbedder` 协议可插拔；混合模型索引/查询触发 `RetrievalNotReadyError`(409)。
- 回归：原有 31 个用例不变，全量 36 passed。
