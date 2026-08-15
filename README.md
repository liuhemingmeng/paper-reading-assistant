# Paper Reading Assistant

一个面向论文和行业研报 AI 阅读助手的学习型项目。本周先完成 Python 工程底座：环境管理、文件与数据处理、HTTP API 调用、重试、命令行参数，以及一个 JSON 持久化的 Todo CLI。

后续路线会逐步加入 FastAPI、PDF 解析、LLM 调用、向量检索、RAG、Agent、评测和 Docker 部署。

## 当前完成内容

- Python 3.11+ 项目配置，当前开发环境使用 Python 3.13。
- 五个标准库练习脚本：
  - `scripts/file_batch.py`：安全的文件批处理预览与执行。
  - `scripts/data_io.py`：JSON/CSV 读写。
  - `scripts/api_client.py`：调用 JSON API 并保存响应。
  - `scripts/retry_demo.py`：有限次数重试与指数退避。
  - `scripts/cli_args.py`：`argparse` 命令行参数解析。
- `todo-cli`：
  - `add` 添加任务。
  - `list` 查看任务。
  - `done` 完成任务。
  - `delete` 删除任务。
  - JSON 文件持久化。
  - 日志与错误处理。
- JSON Todo CLI 与 7 个 pytest 测试，覆盖存储读写和 Todo 生命周期。
- FastAPI + SQLite 论文管理 API：
  - 创建、分页查询、单篇查询、局部更新和删除论文元数据。
  - Pydantic v2 请求/响应校验。
  - SQLAlchemy ORM 与 SQLite 文件数据库。
  - FastAPI TestClient 接口测试。
- PDF 解析与阅读洞见 API（第 3 周）：
  - 受限 PDF 上传、UUID 文件名保存、10 MiB 上限与无效文件清理。
  - PyMuPDF 逐页文本提取、页码/章节可追溯分块。
  - 文档、文本块与阅读洞见的 SQLite 持久化。
  - OpenAI 兼容 LLM 客户端：结构化摘要、5 个问题、有限重试和配置错误保护。
  - Fake LLM、上传失败和重传替换等完整流程测试。
- 本地 RAG 检索与可引用问答（第 4 周）：
  - 以可替换的 `TextEmbedder` 协议隔离 embedding 实现。
  - 内置无依赖、可复现的 hashing vector 基线，用于建立 SQLite chunk 索引。
  - Top-K 余弦检索返回文本、相似度、页码和章节引用。
  - 问答仅使用检索证据，并把同一批证据作为 citations 返回。
- RAG 检索评测（第 5 周）：
  - 以“问题 → 目标证据页码”构造固定、可审计的评测集。
  - 提供逐题命中明细、Recall@K 与 MRR 指标。
  - 可离线回归检索质量，不依赖 LLM 密钥或网络。
- 可替换的真实 Embedding（第 6 周）：
  - 在 `TextEmbedder` 协议下新增 OpenAI 兼容 embedding 客户端（`POST /embeddings`）。
  - `get_default_embedder()` 按 `EMBEDDING_*` 环境变量在真实模型与本地 hashing 基线间切换。
  - 索引与查询共用同一嵌入器，`retrieve_chunks` 对混合模型返回 409 并要求重建索引。
- 答案质量评测（第 7 周）：
  - Citation Correctness（离线）：抽取答案提及的页码，校验是否都属于检索证据页，识别凭空引用。
  - Faithfulness（可选 LLM 裁判）：判断答案是否仅基于检索证据，未配置时明确跳过。
  - 端到端报告同时包含 Recall@K、MRR、引用正确率与忠实率，可在同一固定评测集上对比。
  - `scripts/evaluate_answers.py` 提供命令行评测，`--fake` 无需任何 LLM 凭据即可演示离线链路。
- 接入真实 Embedding（第 8 周）：
  - `OpenAICompatibleEmbedder` 新增可配置 `EMBEDDING_ENDPOINT`，并适配火山方舟多模态端点 `/embeddings/multimodal` 的输入（`input` 为结构化对象数组）与响应（`data` 为单对象）。
  - `EMBEDDING_*` 三变量齐全时切换真实模型；新增 `EMBEDDING_ENDPOINT`（默认 `/embeddings`）支持非标准端点。
  - 测试隔离：让 `client` fixture 显式注入离线 `LocalHashingEmbedder`，并新增多模态解析回归测试，避免真实 `.env` 污染测试。
  - 已用火山方舟 `doubao-embedding-vision-251215` 完成真实联网调用，返回 2048 维归一化向量。
  - 检索质量对比实验（同周）：`scripts/compare_embeddings.py` 用 6 页合成多主题文档 + 12 条问句（字面/语义两类）跑受控基准；火山语义模型 `Recall@K=1.0`、`MRR=1.0`，本地哈希基线 k≥3 也能 100% 召回但 k=1 `MRR=0.944`（语义类 0.833），证明 MRR 比 Recall@K 更能暴露 top-1 质量差距。
  - 评测语料库（第 8 周扩展）：由 5 个子代理并行搜集 **138 篇**真实公开文档——arXiv 三领域（检索/RAG 26、LLM/Agent 30、Embedding 25）共 81 篇，行业资料（金融/招采 29、技术/标准/法规 28）共 57 篇；129 篇已下载、9 篇因登录墙/免责页记为 `link_only`。`scripts/build_corpus_manifest.py` 汇总为 `data/corpus/corpus_manifest.json`，供后续多模型 + reranker 检索基准使用。
- 多模型检索评测底座（第 9 周）：
  - 新增 `Reranker` 协议与 `SiliconFlowReranker`（`POST /v1/rerank`），与 `TextEmbedder` 协议对称，为两阶段检索（embedding 召回 + reranker 重排）做准备；含 3 次指数退避重试、错误边界与按分数降序的 `(index, score)` 解析（兼容 `relevance_score` 与 `score` 两种字段名）。
  - 新增 `model_registry.py`：集中登记 5 个 embedding + 3 个 reranker 的端点与密钥（密钥只读 `.env`，不硬编码），供后续语料级基准按短名实例化。
  - `scripts/smoke_models.py` 联网冒烟 9 个模型：**8/9 可达**——火山 `doubao-embedding-vision-251215`(2048 维)、硅基 `Qwen/Qwen3-VL-Embedding-8B`(4096 维)/`Qwen/Qwen3-Embedding-0.6B`(1024 维)/`BAAI/bge-m3`(1024 维)，以及 3 个硅基 reranker（`Qwen3-VL-Reranker-8B`/`Qwen3-Reranker-4B`/`Qwen3-Reranker-0.6B`）全部正常；火山 `doubao-embedding-large-text-250515` 返回 404 `InvalidEndpointOrModel.NotFound`（该模型在火山模型目录中状态为 `Retiring`/未对当前密钥开通）。已同步修复 embedding 与 rerank 客户端的错误透出，让重试失败时直接显示真实状态码与厂商报错，而非笼统的“重试 3 次失败”。

## 环境要求

- Python 3.11 或更高版本
- Git
- Windows、macOS 或 Linux

## 安装

```bash
git clone https://github.com/liuhemingmeng/paper-reading-assistant.git
cd paper-reading-assistant

python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e .
```

如果 PowerShell 禁止执行激活脚本，可以直接使用 `.venv\\Scripts\\python.exe` 运行命令，不需要激活环境。

## Todo CLI 使用

查看帮助：

```bash
python -m todo_cli --help
python -m todo_cli add --help
```

添加、查看、完成和删除：

```bash
python -m todo_cli add "Read a RAG paper"
python -m todo_cli list
python -m todo_cli done 1
python -m todo_cli delete 1
```

默认数据文件为 `data/todos.json`。也可以指定其他路径，便于测试：

```bash
python -m todo_cli --data-file data/demo.json add "Learn pathlib"
python -m todo_cli --data-file data/demo.json list
```

## 论文管理 API（第 2 周）

启动开发服务器：

```bash
python -m paper_api
```

服务默认运行在 `http://127.0.0.1:8000`。启动后可打开：

- 交互式 API 文档：`http://127.0.0.1:8000/docs`
- OpenAPI JSON：`http://127.0.0.1:8000/openapi.json`
- 健康检查：`http://127.0.0.1:8000/health`

API 使用 SQLite 数据库 `data/papers.db`，首次启动时会自动创建表。数据库文件是本地生成数据，已被 Git 忽略。

创建论文：

```bash
curl -X POST http://127.0.0.1:8000/papers \
  -H "Content-Type: application/json" \
  -d '{"title":"RAG Survey","authors":"Lewis, Patrick","abstract":"A survey of retrieval-augmented generation."}'
```

查询、更新和删除：

```bash
curl http://127.0.0.1:8000/papers?offset=0\&limit=20
curl http://127.0.0.1:8000/papers/1
curl -X PATCH http://127.0.0.1:8000/papers/1 -H "Content-Type: application/json" -d '{"title":"Updated RAG Survey"}'
curl -X DELETE http://127.0.0.1:8000/papers/1
```

接口契约和字段约束见 [第 2 周 API 设计](docs/week-02-api-design.md)。

## PDF 上传与阅读洞见（第 3 周）

先按上文启动 API。创建一条论文后，使用 `multipart/form-data` 上传 PDF：

```bash
curl -X POST http://127.0.0.1:8000/papers/1/document \
  -F "file=@./example.pdf;type=application/pdf"
```

上传成功后，服务会把文件保存至 `data/uploads/paper-1/`，提取每页文本并建立可追溯的文本块。上传文件、SQLite 数据库以及真实 `.env` 都是本地状态，已被 Git 忽略。

```bash
curl http://127.0.0.1:8000/papers/1/document
curl http://127.0.0.1:8000/papers/1/chunks?offset=0\&limit=100
```

PDF 当前必须满足以下限制：仅 `.pdf` 后缀、MIME 类型为 `application/pdf`（或缺失/通用二进制类型）、文件不超过 10 MiB，且必须包含可提取的文本层。扫描件需要 OCR，属于后续阶段而非本周范围。

### 配置真实 LLM（可选）

复制模板后，只在本机 `.env` 填入 OpenAI 兼容服务的地址、密钥和模型名：

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS/Linux
cp .env.example .env
```

```text
LLM_BASE_URL=https://api.example.com/v1
LLM_API_KEY=replace-with-your-secret
LLM_MODEL=replace-with-your-model-name
```

随后可请求结构化的摘要与 5 个阅读问题：

```bash
curl -X POST http://127.0.0.1:8000/papers/1/insights:generate
curl http://127.0.0.1:8000/papers/1/insight
```

未配置这些变量时，生成接口会明确返回 `503`，不会把请求发送到未知服务。具体数据模型与边界见 [第 3 周 PDF/LLM 设计](docs/week-03-pdf-llm-design.md)。

## RAG 检索与引用问答（第 4 周）

PDF 上传完成后先显式建立本地索引，再检索问题相关的文本块：

```bash
curl -X POST http://127.0.0.1:8000/papers/1/retrieval:index
curl "http://127.0.0.1:8000/papers/1/search?query=How%20does%20retrieval%20work%3F&limit=3"
```

搜索响应包含文本块、相似度、页码与章节名。当前使用本地 hashing vector 作为可复现学习基线，适合理解完整 RAG 流程；它不是语义 embedding 模型，后续将替换为真实 embedding 服务或模型。

配置 LLM 后可生成只基于检索证据的回答：

```bash
curl -X POST "http://127.0.0.1:8000/papers/1/questions:answer?question=How%20does%20retrieval%20work%3F&limit=3"
```

响应中的 `citations` 与送入模型的证据一一对应。未建索引返回 `409`，空问题或无相关证据返回 `422`，未配置 LLM 时问答返回 `503`。设计取舍见 [第 4 周 RAG 设计](docs/week-04-rag-design.md)。

## RAG 检索评测（第 5 周）

在索引完成后，以人工标注的“问题 → 目标证据页码”运行评测。接口返回每题的检索页码、命中结果和倒数排名，并汇总 `Recall@K` 与 `MRR`：

```bash
curl -X POST "http://127.0.0.1:8000/papers/1/retrieval:evaluate?k=3" \
  -H "Content-Type: application/json" \
  -d '[
    {"question":"What finds relevant evidence?","expected_page_numbers":[1]},
    {"question":"What measures ranking quality?","expected_page_numbers":[2]}
  ]'
```

`Recall@K` 表示有多少问题在前 K 个结果中找到了至少一个目标页；`MRR` 额外关注第一个目标页的排名。该评测先覆盖检索质量，回答忠实度仍需后续人工标注或裁判模型评测。详细定义见 [第 5 周评测设计](docs/week-05-evaluation-design.md)。

## 可替换 Embedding（第 6 周）

检索索引默认使用本地、无依赖的 hashing 向量。配置以下环境变量后，`create_app` 会改用真实 embedding 模型，索引与查询随之切换：

```bash
EMBEDDING_BASE_URL=https://your-endpoint/v1
EMBEDDING_API_KEY=replace-me
EMBEDDING_MODEL=your-embedding-model
```

切换真实模型后，需要对该论文重新执行 `POST /papers/{id}/retrieval:index` 重建索引，再用第 5 周同一份问题集运行 `retrieval:evaluate`，即可对比本地基线与新模型的 `Recall@K` / `MRR`。配置、重试、错误边界与第 3 周 LLM 客户端保持一致；未配置时返回 `409`（未索引）而非误调用外部服务。设计细节见 [第 6 周 Embedding 设计](docs/week-06-embedding-design.md)。

## 基础脚本示例

文件批处理默认只预览，不修改文件：

```bash
python scripts/file_batch.py data --prefix demo
python scripts/file_batch.py data --prefix demo --apply
```

JSON/CSV 读写：

```bash
python scripts/data_io.py --output data/sample.json
python scripts/data_io.py --output data/sample.csv
```

命令行参数：

```bash
python scripts/cli_args.py RAG --pages 3 --tag python --tag learning
```

重试示例：

```bash
python scripts/retry_demo.py --attempts 3
```

API 示例需要真实的 JSON API 地址。先复制环境变量模板：

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS/Linux
cp .env.example .env
```

也可以直接传 URL：

```bash
python scripts/api_client.py https://httpbin.org/json --output data/api_response.json
```

不要把真实 API 密钥提交到 Git。`.env` 已被 `.gitignore` 忽略。

## 测试

```bash
python -m pytest -q
```

当前预期结果：`61 passed`。测试覆盖 Todo CLI、论文 CRUD、PDF 上传/解析/分块/重传替换/删除清理、本地向量索引、Top-K 引用检索、RAG 回答边界、Recall@K/MRR 评测指标、可替换 embedding 客户端与混合模型防护，第 7 周的引用正确性、忠实度裁判与端到端答案评测路由，第 8 周的多模态 embedding 解析回归，第 9 周的 reranker 客户端（payload 构造、响应解析、容错与配置校验）与基准相关单测（BM25 排序、IR 指标正确性）。

## 教程

每个完成的项目阶段都会在 `docs/tutorials/` 产出一份 HTML 教程，讲解真实实现、设计原因、常见错误、验证方法和后续迁移路径。

- [第 1 周：Python 工程底座教程](docs/tutorials/week-01-python-foundation.html)
- [第 2 周：FastAPI + SQLite 论文管理 API 教程](docs/tutorials/week-02-fastapi-sqlite.html)
- [第 3 周：PDF 解析与 LLM 阅读洞见教程](docs/tutorials/week-03-pdf-llm.html)
- [第 4 周：本地 RAG 检索与可引用问答教程](docs/tutorials/week-04-rag-retrieval.html)
- [第 5 周：RAG 检索与引用评测教程](docs/tutorials/week-05-rag-evaluation.html)
- [第 6 周：可替换真实 Embedding 教程](docs/tutorials/week-06-embedding.html)
- [第 7 周：答案质量评测教程](docs/tutorials/week-07-answer-evaluation.html)
- [第 8 周：接入真实 Embedding（火山多模态）教程](docs/tutorials/week-08-embedding-config.html)
- [第 8 周：本地基线 vs 火山语义向量 检索对比实验报告](docs/tutorials/week-08-embedding-experiment.html)
- [第 9 周：多模型注册表与 reranker 客户端教程](docs/tutorials/week-09-multimodel-reranker.html)
- [第 9 周：129 篇真实语料检索基准（6 检索器 × 3 reranker）教程](docs/tutorials/week-09-retrieval-benchmark.html)

也可以检查所有 Python 文件是否能编译：

```bash
python -m compileall -q scripts src tests
```

## 目录结构

```text
.
├── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── scripts/
├── src/
│   ├── todo_cli/
│   └── paper_api/
├── tests/
├── docs/
│   ├── tutorials/
│   ├── week-03-pdf-llm-design.md
│   ├── week-04-rag-design.md
│   ├── week-05-evaluation-design.md
│   ├── week-06-embedding-design.md
│   └── week-07-answer-evaluation-design.md
└── data/
```

项目使用 `src/` 布局。执行 `python -m pip install -e .` 后，Python 才会把 `src/todo_cli` 作为可导入包使用；测试通过 `pyproject.toml` 中的 `pythonpath` 配置加载源码。

## 本周学习重点

- 用 `pathlib` 表达文件路径，而不是手写平台相关字符串。
- 用类型注解、数据类和小函数划分模块职责。
- 区分用户可见输出、异常和 `logging` 日志。
- 用环境变量保存配置，用 `.env.example` 记录变量名，避免提交秘密。
- 用有限重试和指数退避处理临时网络故障。
- 用 pytest 和临时目录测试文件存储，避免污染真实数据。
- 用小而清晰的 Git 提交记录工程演进过程。
- 用 `UploadFile` 分块落盘，而不是将任意文件一次性读入内存。
- 用页码、章节名、序号维护文本来源，给后续 RAG 回答提供可引用证据。
- 将外部 LLM API 封装为可替换客户端，并把配置、重试、响应校验放在边界层。
- 区分 embedding、索引、检索、生成四个 RAG 阶段，而不是把它们混成一次模型调用。
- 使用余弦相似度排序 Top-K evidence，并在最终回答中原样返回 citations。
- 用接口协议和 FakeGenerator 隔离外部模型，使检索和引用链路可离线测试。
- 用固定问题和目标证据页码建立可复现的 retrieval evaluation set。
- 区分“前 K 个结果是否命中”的 Recall@K 与“第一个命中排第几”的 MRR。
- 用逐题结果审计汇总指标，避免只看一个看似漂亮的平均分。
- 用 `TextEmbedder` 协议隔离 embedding 实现，让本地基线与真实模型共享同一套检索与评测代码。
- 让索引与查询共用一个嵌入器，并对混合模型返回明确错误，避免无意义的相似度。

## 下一步

第 8 周已完成真实 Embedding 接入、检索质量对比实验，并建成 138 篇真实公开评测语料库（详见 `docs/tutorials/week-08-embedding-experiment.html` 与 `data/corpus/README.md`）。第 9 周已落地 `Reranker` 协议 + `SiliconFlowReranker` 客户端、`model_registry.py` 多模型登记、`scripts/smoke_models.py` 联网冒烟（8/9 模型可达；火山 `doubao-embedding-large-text-250515` 因 `Retiring`/未开通而 404，已记录并修复错误透出），并完成了**语料级检索基准**：`src/paper_api/corpus.py`（文本抽取）、`retrieval.py` 的 `BM25Retriever`、`ir_metrics.py`（Recall@K/MRR/nDCG@K）、`scripts/benchmark_retrieval.py`。在 129 篇真实文档（arXiv 79 + 行业 50）上对比 6 个基础检索器 × 3 个 reranker，**全部 18 个组合 errors=0**。关键结论：bge-m3 / qwen3-embed-0.6b 文本检索最强（R@1 0.953 / 0.922）；VL-embedding（qwen3vl-embed-8b）不适合纯文本检索（arXiv R@1 仅 0.241）；reranker 的真正价值是救回弱召回（local-hashing 0.333→0.86、qwen3vl-embed 0.426→0.89），接在强检索器后几乎无增益甚至轻微负向。详见 `docs/tutorials/week-09-multimodel-reranker.html` 与 `docs/tutorials/week-09-retrieval-benchmark.html`。

下一步（生产级闭环）：① 构造「自然语言问题→证据段落」真实问答对重跑基准，才是 reranker 的主场（已知项检索低估其价值）；② 接入真实 `LLM_*` 用 `scripts/evaluate_answers.py --faithfulness` 跑答案质量闭环；③ 对 embedding 维度、reranker 候选数、chunk 大小做消融。当前离线 Fake 链路仍保证每次提交都能回归检索命中与引用正确性，不依赖任何外部凭据。
