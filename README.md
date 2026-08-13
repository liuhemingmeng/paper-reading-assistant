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

当前预期结果：`24 passed`。测试覆盖 Todo CLI、论文 CRUD、PDF 上传/解析/分块/重传替换/删除清理，以及 Fake LLM 洞见持久化。

## 教程

每个完成的项目阶段都会在 `docs/tutorials/` 产出一份 HTML 教程，讲解真实实现、设计原因、常见错误、验证方法和后续迁移路径。

- [第 1 周：Python 工程底座教程](docs/tutorials/week-01-python-foundation.html)
- [第 2 周：FastAPI + SQLite 论文管理 API 教程](docs/tutorials/week-02-fastapi-sqlite.html)
- [第 3 周：PDF 解析与 LLM 阅读洞见教程](docs/tutorials/week-03-pdf-llm.html)

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
│   └── week-03-pdf-llm-design.md
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

## 下一步

第 4 周将在 `PaperChunk` 上构建向量索引和检索链路，再让 RAG 回答携带来源页码。此时第 3 周保存的分块、序号与页码就是检索和可解释回答的基础。
