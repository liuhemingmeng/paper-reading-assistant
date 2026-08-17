# Paper Reading Assistant

[![CI](https://github.com/liuhemingmeng/paper-reading-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/liuhemingmeng/paper-reading-assistant/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于 RAG 的论文与行业研报 AI 阅读助手：从 PDF 上传、语义分块、向量检索到**带引用的问答生成**，并配套一套可复现的**检索 / 答案质量评测闭环**。支持单篇论文与跨语料库两种检索域，生产环境可接入 PostgreSQL + pgvector 做 ANN 加速。所有外部模型走标准 OpenAI 兼容协议，更换厂商或自托管只需改 `.env` 与模型注册表，业务代码零改动。

## Features

- **PDF 全流程处理**：上传（类型 / 大小校验）→ PyMuPDF 逐页解析 → 滑动窗口分块，保留页码 / 章节 / 序号溯源。
- **双检索域**：单篇论文检索 + 跨语料库检索，citations 携带论文标题。
- **可替换 Embedding**：`TextEmbedder` 协议隔离实现，默认 `BAAI/bge-m3`，内置零依赖离线 `local-hashing` 基线；索引与查询强制使用同一模型（混合模型返回 409）。
- **可选 Reranker**：交叉编码器重排（qwen3-rerank 系列），默认关闭；在弱检索器上可显著救回召回。
- **带引用的 RAG 问答**：答案仅基于检索证据生成，证据以 `citations` 原样返回（标题 / 页码 / 章节 / 得分）。
- **评测闭环**：Recall@K / MRR / Citation Correctness / Faithfulness，固定评测集可离线回归，不依赖外部凭据。
- **鉴权与密钥轮换**：`X-API-Key` 可选鉴权，支持运行时轮换（旧 key 立即失效、无需重启）。
- **双存储后端**：同一套代码本地走 SQLite（开发 / 测试），生产走 pgvector HNSW ANN。
- **工程化**：单页 chat UI、Docker 镜像、GitHub Actions CI、67 项自动化测试。

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    前端 chat UI（frontend/ 单页）                   │
│       问答模式（生成带引用答案） / 检索模式（仅看证据） / 密钥设置       │
└──────────────────────────────┬──────────────────────────────────┘
                               │  HTTP + X-API-Key（可选） · X-Admin-Key（轮换）
┌──────────────────────────────▼──────────────────────────────────┐
│                 FastAPI（src/paper_api/api.py）                   │
│   跨语料库：GET /corpus/search · POST /corpus/questions:answer     │
│   单篇：    GET /papers/{id}/search · POST /papers/{id}/questions:answer │
│   评测：    /papers/{id}/retrieval:evaluate · /answers:evaluate    │
│   管理：    POST /admin/rotate-key（运行时密钥轮换）                │
└──────────────┬───────────────────────────────┬──────────────────┘
               │ services.py（检索+生成编排）      │ OpenAI 兼容协议
┌──────────────▼──────────────────┐   ┌─────────▼──────────────────┐
│ 存储层（双后端，同一套代码）        │   │ 外部模型（注册表可替换）      │
│  PostgreSQL + pgvector（生产）    │   │  Embedding: bge-m3 (R@1=0.953)│
│    HNSW ANN · 余弦距离 · 向量表    │   │  Reranker: qwen3-rerank（可选）│
│  SQLite（本地开发/测试自动回退）    │   │  LLM: deepseek-v4-flash      │
└──────────────────────────────────┘   └────────────────────────────┘
```

## Quickstart

```bash
git clone https://github.com/liuhemingmeng/paper-reading-assistant.git
cd paper-reading-assistant

python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux:      source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e .

cp .env.example .env   # 填入你申请的模型密钥（可选，不填也能跑离线 hashing 基线）
python -m paper_api    # 启动服务，默认 http://127.0.0.1:8000
```

打开前端 chat UI：<http://127.0.0.1:8000/insight>

- **问答模式**：向语料库提问，答案附带来源引用。
- **检索模式**：只看命中的证据，不调用 LLM。
- **密钥设置**：在页面内填写 Embedding / LLM 密钥，保存后即时生效，无需重启。

预置演示语料（3 篇样例论文，覆盖 RAG / Attention / VectorDB，索引已建好）：

```bash
python seed_demo.py    # 需要服务已启动；仅首次执行
```

## Usage

跨语料库检索（无需 LLM）：

```bash
curl "http://127.0.0.1:8000/corpus/search?query=neural%20IR&limit=3"
```

跨语料库问答（需要 LLM 配置，`citations` 携带 `paper_title`）：

```bash
curl -X POST "http://127.0.0.1:8000/corpus/questions:answer?question=What%20is%20neural%20IR%3F&limit=5"
```

上传论文并建立检索索引：

```bash
# 1) 创建论文
curl -X POST http://127.0.0.1:8000/papers \
  -H "Content-Type: application/json" \
  -d '{"title":"RAG Survey","authors":"Lewis, Patrick","abstract":"A survey of retrieval-augmented generation."}'

# 2) 上传 PDF（multipart）
curl -X POST http://127.0.0.1:8000/papers/1/document -F "file=@paper.pdf;type=application/pdf"

# 3) 建索引（索引与查询必须使用同一 embedding 模型）
curl -X POST http://127.0.0.1:8000/papers/1/retrieval:index
```

单篇检索 / 问答 / 评测：

```bash
curl "http://127.0.0.1:8000/papers/1/search?query=How%20does%20retrieval%20work%3F&limit=3"
curl -X POST "http://127.0.0.1:8000/papers/1/questions:answer?question=How%20does%20retrieval%20work%3F&limit=3"
curl -X POST "http://127.0.0.1:8000/papers/1/retrieval:evaluate?k=3" \
  -H "Content-Type: application/json" \
  -d '[{"question":"What finds relevant evidence?","expected_page_numbers":[1]}]'
```

运行时轮换访问密钥（旧 key 立即失效，无需重启）：

```bash
curl -X POST http://127.0.0.1:8000/admin/rotate-key -H "X-Admin-Key: <你的 ADMIN_KEY>"
```

## API Overview

| Method | Path | 说明 |
|---|---|---|
| POST | `/papers` | 创建论文 |
| GET | `/papers?offset=&limit=` | 分页查询论文 |
| PATCH / DELETE | `/papers/{id}` | 更新 / 删除论文 |
| POST / GET | `/papers/{id}/document` | 上传 / 查询 PDF 文档 |
| GET | `/papers/{id}/chunks` | 查看分块 |
| POST | `/papers/{id}/retrieval:index` | 建立检索索引 |
| GET | `/papers/{id}/search` | 单篇检索 |
| POST | `/papers/{id}/questions:answer` | 单篇问答（带引用） |
| GET | `/corpus/search` | 跨语料库检索 |
| POST | `/corpus/questions:answer` | 跨语料库问答（带引用） |
| POST | `/papers/{id}/retrieval:evaluate` | 检索评测（Recall@K / MRR） |
| POST | `/papers/{id}/answers:evaluate` | 答案评测（引用正确率 / 忠实率） |
| GET | `/settings/status` | 当前模型配置（不含密钥） |
| POST | `/settings` | 更新模型密钥（即时生效） |
| POST | `/admin/rotate-key` | 轮换访问密钥（需 `X-Admin-Key`） |
| GET | `/insight` | 前端 chat UI |

未设置 `RAG_API_KEY` 时接口开放（本地开发 / 测试默认）；设置后所有数据接口要求 `X-API-Key` 请求头。

## Retrieval Benchmarks

在 129 篇真实 arXiv / 行业文档上做文档级检索基准，指导生产选型：

| 配置 | R@1 | 结论 |
|---|---|---|
| **bge-m3（生产默认）** | **0.953** | 文本检索最强，中英文均衡，1024 维 |
| qwen3-embed-0.6b | 0.922 | 备选文本模型 |
| qwen3vl-embed-8b（视觉） | 0.241 | 不适合纯文本检索，仅实验 |
| local-hashing + qwen3-rerank-4b | 0.333 → 0.86 | reranker 救回弱检索器 |
| bge-m3 + qwen3-rerank-4b | −0.016 | 接在强检索器后几乎无增益，默认不接 |

生产配置：**bge-m3 + 默认不接 reranker + deepseek-v4-flash-260425 + 1200/160 滑动窗口分块（句子边界优先）+ top-k 证据（默认 3，前端 5）**。

## Design Decisions

- **为什么 reranker 默认不接**：reranker 的价值在于救回弱检索器（0.333 → 0.86）；接在 bge-m3 这类强检索器之后增益趋近于零（−0.016），却增加一次外部调用延迟与成本。作为可选项保留，`RERANKER_ENABLED=true` 一键开启。
- **为什么双存储后端**：本地开发与 CI 用 SQLite（零配置、可离线），生产用 pgvector（HNSW ANN、毫秒级近邻查询、与业务数据同库事务）。通过 SQLAlchemy 方言探测在启动时选择向量表策略，同一套检索代码零分支。
- **为什么 Embedding 可替换**：`TextEmbedder` 协议隔离实现细节，本地 hashing 基线保证零密钥也能跑通全链路，真实模型通过 `EMBEDDING_*` 环境变量切换。索引与查询强制同模型，混合模型直接返回 409 而非产出无意义相似度。
- **为什么分块带句子边界与章节溯源**：PDF 上传按 1200 字符 / 160 重叠滑窗切分，窗口末尾优先回退到句号 / 换行切断，避免把句子劈成两半；章节标题被识别为 section_title 供 citations 引用（语料库基准则用 900/120 纯滑窗并携带 doc_id）。
- **为什么密钥默认内置但永不入库**：本地 `.env` 内置真实演示密钥实现零配置一键运行，`.env` 被 gitignore 且 GitHub push protection 会拦截密钥提交；`.env.example` 只保留变量名模板。
- **为什么鉴权可选**：本地演示与 CI 默认开放（`RAG_API_KEY` 留空），生产通过容器环境注入 `RAG_API_KEY` 启用；`/insight`、静态资源、健康检查始终免鉴权，保证页面可被直接打开。
- **为什么支持运行时密钥轮换**：`verify_api_key` 每次请求实时读取环境变量，因此轮换端点只需更新 `os.environ` 并落盘 `.env`，旧 key 当场失效、无需重启容器；轮换动作本身由独立 `ADMIN_KEY` 保护。

## Deployment

生产环境为 Docker 容器 + Nginx 反代，参考 `Dockerfile`：

```bash
docker build -t rag-app:latest .
# 密钥通过 --env-file 注入，不进镜像
docker run -d --name rag-app --env-file /opt/rag/.env \
  -p 127.0.0.1:8000:8000 --network rag_default \
  --restart unless-stopped rag-app:latest
```

要点：

- **pgvector 持久化**：容器内 PostgreSQL + pgvector（命名卷 `rag_pgvector_data`），5432 不暴露公网，仅容器内网可达。
- **Nginx 反代**：只开 80 端口，`/` 反代到 8000，透传 `X-API-Key`，配 `limit_req` 限流（超限返回 429）。
- **鉴权**：容器注入 `RAG_API_KEY` 后所有数据接口需带 `X-API-Key`。
- **密钥轮换**：容器 `.env` 中配置 `ADMIN_KEY`，随时调用 `/admin/rotate-key` 换发新 key 并收回旧 key 权限。
- **gunicorn**：`-w 2 --preload`，避免多 worker 并发建表竞态。
- 密钥用完即弃：演示结束后在模型服务商后台统一禁用即可。

## Project Structure

```text
.
├── README.md
├── .env.example          # 环境变量模板（不含任何真实密钥）
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yml
├── seed_demo.py          # 预置 3 篇演示论文（自动建索引）
├── src/
│   ├── paper_api/        # 核心服务
│   │   ├── api.py        # FastAPI 路由（单篇/跨库检索、问答、评测、设置、轮换）
│   │   ├── services.py   # 检索 + 生成编排
│   │   ├── vector_store.py   # pgvector ANN（SQLite 自动回退）
│   │   ├── auth.py       # API Key 鉴权 + Admin Key
│   │   ├── embeddings.py # TextEmbedder 协议 + OpenAI 兼容客户端 + 本地基线
│   │   ├── rerank.py     # 交叉编码器 reranker（可选）
│   │   ├── llm_client.py # OpenAI 兼容 LLM 客户端
│   │   ├── chunking.py   # 滑动窗口分块（1000/120）
│   │   ├── pdf_processing.py  # PyMuPDF 解析 + 上传校验
│   │   ├── evaluation.py / answer_evaluation.py / ir_metrics.py  # 评测
│   │   ├── model_registry.py  # 模型注册表
│   │   └── settings.py   # .env 读写
│   └── todo_cli/         # 早期练习 CLI（保留，独立于主服务）
├── frontend/
│   └── index.html        # 单页 chat UI
├── scripts/              # 基准/冒烟/语料构建脚本
└── tests/                # 67 项 pytest
```

## Testing

```bash
python -m pytest -q
```

预期 `67 passed`。测试覆盖论文 CRUD、PDF 上传 / 解析 / 分块 / 重传替换、本地索引、Top-K 引用检索、RAG 回答边界、Recall@K / MRR / 引用正确率 / 忠实率评测、embedding / reranker / LLM 客户端（payload 构造、响应解析、容错、配置校验）、混合模型 409 防护、跨语料库检索。CI 中显式清空 `EMBEDDING_*` / `LLM_*` 环境变量，保证测试套件完全离线、可复现。

## License

[MIT](LICENSE)
