# 第 4 周设计：本地向量检索与可引用 RAG 问答

## 目标

基于第 3 周持久化的 `PaperChunk`，构建一个可运行、可测试的 RAG 最小闭环：为文本块建立向量索引，按问题检索 Top-K 证据，再仅将这些证据交给 LLM 生成回答，并把页码与章节作为 citations 返回。

## 为什么本周不用云向量数据库或真实 embedding API

- 当前目标是理解 RAG 数据流、接口契约、相似度排序和引用，而不是绑定某个供应商。
- `LocalHashingEmbedder` 完全本地运行，测试确定、无需密钥或下载模型。
- 它是工程训练用的 lexical baseline，不是语义 embedding 模型；生产或项目后期可实现同一个 `TextEmbedder` 协议来替换为真实模型。
- 向量先以 JSON 存入 SQLite，避免第 4 周同时引入数据库部署与 ANN 索引复杂度。

## 数据模型

新增 `ChunkEmbedding`：

| 字段 | 用途 |
| --- | --- |
| chunk_id | 与 `PaperChunk` 一一对应 |
| model | 当前嵌入器版本，例如 `local-hashing-v1` |
| dimensions | 向量维度，用于兼容性检查 |
| vector_json | 归一化向量 JSON |
| created_at | 建索引时间 |

重传 PDF 时原 `PaperDocument` 的级联删除会删除旧 chunks，也会沿关系删除 embeddings；索引不会跨文件版本污染。

## 路由

| 方法 | 路径 | 成功 | 说明 |
| --- | --- | --- | --- |
| POST | `/papers/{paper_id}/retrieval:index` | 200 | 为当前 chunks 生成或重建本地向量索引 |
| GET | `/papers/{paper_id}/search?query=...&limit=3` | 200 | 返回排序后的 evidence chunks，含 score、页码与章节 |
| POST | `/papers/{paper_id}/questions:answer?question=...&limit=3` | 200 | 检索后仅使用 citations 生成 grounded answer |

未上传文档返回 404；尚未索引返回 409；空问题或无匹配证据返回 422；缺少 LLM 配置返回 503；模型网络或响应错误返回 502。

## 检索流程

1. 解析 PDF 后已有 `PaperChunk(content, page_number, section_title, sequence)`。
2. `retrieval:index` 调用 `TextEmbedder.embed(chunk.content)`，保存归一化向量。
3. `search` 调用同一 embedder 生成 query vector。
4. 对每个 chunk 计算余弦相似度。由于向量已归一化，余弦相似度等于点积。
5. 以 `score` 降序、`sequence` 升序排序，取得 Top-K。
6. `questions:answer` 将每个证据格式化为 `[chunk:id; page:n; section:x]`，再调用 LLM。
7. API 返回模型答案以及同一批 citations，客户端可展示原文和页码。

## 提示词边界

问答客户端的 system prompt 要求：仅根据 supplied evidence 回答；证据不足时明确说明；禁止补充未在证据中出现的事实。代码层无法保证模型完全不幻觉，因此 citations 是产品层的必要输出，而不是可选装饰。

## 测试范围

- 建立索引前搜索返回 409。
- 建立索引后关键词查询命中正确页码、章节和正分数。
- Fake AnswerGenerator 断言只接收到 Top-1 evidence，证明完整文档不会直接进入问答。
- 空 query 和无关 query 返回 422。
- 已索引但未配置 LLM 时问答返回 503。
- 保留之前 PDF、重传、删除清理与洞见测试，防止管道回归。

## 已知限制与第 5 周方向

当前哈希向量偏向词面重合，不能稳定理解同义词、跨语言表达或复杂语义。SQLite 上的线性扫描适合单文档学习项目，不适合大规模语料。下一阶段可接入真实 embedding、向量索引、检索评测集、答案引用格式校验，并考虑异步建索引任务。
