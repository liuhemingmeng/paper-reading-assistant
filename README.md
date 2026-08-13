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

当前预期结果：`7 passed`。

## 教程

每个完成的项目阶段都会在 `docs/tutorials/` 产出一份 HTML 教程，讲解真实实现、设计原因、常见错误、验证方法和后续迁移路径。

- [第 1 周：Python 工程底座教程](docs/tutorials/week-01-python-foundation.html)
- [第 2 周：FastAPI + SQLite 论文管理 API 教程](docs/tutorials/week-02-fastapi-sqlite.html)

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
│   └── todo_cli/
├── tests/
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

## 下一步

第 2 周将把 Todo CLI 的工程基础迁移到论文管理场景，使用 FastAPI 和 SQLite 实现论文条目的增删改查 API，并补充接口测试。
