# 第 2 周 API 设计：论文管理服务

## 目标

实现一个可运行、可测试的论文管理后端，为第 3 周 PDF 解析和第 4 周 RAG 建立稳定的数据入口。

## 技术选择

- FastAPI：HTTP 路由、请求解析、自动 OpenAPI 文档。
- Pydantic：请求与响应数据校验。
- SQLAlchemy ORM：Python 对象与 SQLite 表之间的映射。
- SQLite：零配置、单文件数据库，适合学习和本地单用户开发。
- pytest + FastAPI TestClient：无需启动真实服务器即可测试 HTTP 接口。

## 数据模型

Paper 字段：

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| id | integer | 自增主键 | 数据库生成 |
| title | string | 去空格后 1-300 字符 | 论文/研报标题 |
| authors | string | 去空格后 1-500 字符 | 作者，可用逗号分隔 |
| abstract | string | 去空格后 1-10000 字符 | 摘要 |
| file_path | string | 可选，最多 1000 字符 | 后续 PDF 文件的本地/对象存储路径 |
| created_at | datetime | 服务端生成 | 创建时间，UTC |
| updated_at | datetime | 服务端更新 | 最近修改时间，UTC |

## 路由

| 方法 | 路径 | 行为 | 成功状态 |
| --- | --- | --- | --- |
| GET | /health | 健康检查 | 200 |
| POST | /papers | 创建论文 | 201 |
| GET | /papers | 分页查询论文 | 200 |
| GET | /papers/{paper_id} | 查询单篇论文 | 200 |
| PATCH | /papers/{paper_id} | 局部更新论文 | 200 |
| DELETE | /papers/{paper_id} | 删除论文 | 204 |

未找到的论文统一返回 404；请求体校验失败由 FastAPI 返回 422。

## 分层边界

- api.py：HTTP 路由、状态码、依赖注入。
- schemas.py：Pydantic 输入/输出模型和字段校验。
- models.py：SQLAlchemy ORM 表定义。
- database.py：Engine、Session 工厂和建表。
- services.py：论文 CRUD 业务逻辑和 NotFound 异常。

## 测试范围

- 健康检查。
- 创建后返回 201、数据库自动生成 id 和时间。
- 列表分页。
- 单条查询、局部更新和删除。
- 404 与 422 错误场景。
- 每个测试使用独立内存 SQLite 数据库，避免测试污染本地 data/ 目录。
