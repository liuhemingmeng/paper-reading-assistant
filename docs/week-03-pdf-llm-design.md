# 第 3 周设计：PDF 解析与 LLM 阅读洞见

## 目标

让已有论文管理 API 能处理真实 PDF：上传、受控保存、文本提取、按页/段落/章节分块、结构化持久化，并在配置 OpenAI 兼容 LLM 后生成摘要与 5 个可能问题。

## 非目标

- 不实现 OCR。扫描型 PDF 没有文本层时返回可读错误；OCR 属于后续增强。
- 不实现向量检索、RAG 或 Agent；文本块与页码来源是第 4 周的输入。
- 不把真实 API Key、上传的 PDF 或 SQLite 数据库提交到 Git。

## 新增数据模型

| 模型 | 关键字段 | 用途 |
| --- | --- | --- |
| PaperDocument | paper_id、原文件名、存储路径、文件大小、页数、完整文本、状态 | 一篇论文当前上传文档的处理记录 |
| PaperChunk | document_id、sequence、page_number、section_title、content、char_count | 可追溯的原文片段 |
| PaperInsight | paper_id、summary、questions_json、model、status、error_message | LLM 生成的阅读洞见 |

`Paper` 表保持原样；新增表通过外键关联，因此旧的 SQLite 数据库不会因新增字段而失效。

## 路由

| 方法 | 路径 | 行为 | 成功状态 |
| --- | --- | --- | --- |
| POST | /papers/{paper_id}/document | 上传、保存、解析 PDF 并写入分块 | 201 |
| GET | /papers/{paper_id}/document | 获取文档处理元数据 | 200 |
| GET | /papers/{paper_id}/chunks | 查询已提取文本块 | 200 |
| POST | /papers/{paper_id}/insights:generate | 调用 LLM 生成摘要和问题 | 201 |
| GET | /papers/{paper_id}/insight | 获取最近一次阅读洞见 | 200 |

未找到 Paper 或资源时返回 404；上传类型/大小/内容不符合要求返回 400 或 422；无 LLM 配置时洞见生成返回 503；PDF 无文本或解析失败返回 422。

## 文件处理策略

- 仅接受 `.pdf` 文件，最大 10 MiB。
- 使用 `UploadFile` 分块写入 `data/uploads/paper-{id}/`，不把整个文件一次读入内存。
- 使用 UUID 生成存储文件名，原始文件名只作为元数据，避免路径穿越和同名覆盖。
- 使用 PyMuPDF 逐页提取文本，清理空白行；空文本 PDF 不做伪成功处理。
- 文本块最大 1200 字符，段落优先，超过上限时按字符切分并保留 160 字符重叠。
- 每个文本块保存 page_number、sequence 和已识别的 section_title，供第 4 周引用和检索使用。

## LLM 配置

通过本地 `.env` 或环境变量提供：

```text
LLM_BASE_URL=https://api.example.com/v1
LLM_API_KEY=your-secret
LLM_MODEL=your-model-name
```

客户端调用 OpenAI 兼容的 `POST /chat/completions`。请求有连接/读取/写入/连接池超时和有限退避重试；只对超时、连接失败、429 和 5xx 重试。模型必须返回 JSON 对象：`summary` 和恰好 5 条 `questions`。

## 测试

- 测试内动态生成最小有文本 PDF，不依赖外部附件。
- 使用内存 SQLite + StaticPool 隔离每个测试。
- 验证上传成功、分块页码、非 PDF、空文本 PDF、未知 Paper。
- 注入假 LLM 客户端测试洞见落库；不在测试中请求真实网络。
- 验证无 LLM 配置的 503、异常响应和已保存洞见查询。
