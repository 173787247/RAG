# 自定义 RAG 应用 (Custom RAG Application)

本项目是一个基于 LlamaIndex 构建的自定义检索增强生成 (RAG) 应用，允许用户上传文档，并通过大语言模型 (LLM) 对文档内容进行提问和回答。

## 功能特性

*   **文档加载 (Load File)**:
    *   支持从本地文件系统上传单个或多个文档 (.txt, .pdf, .md, .docx)。
    *   支持从 URL 加载网页内容。
*   **高级文件解析 (Parse File)**:
    *   优先使用 LlamaParse 服务（需要 `LLAMA_CLOUD_API_KEY`）对上传的文档（特别是 PDF 和 DOCX）进行高级解析，能够提取表格和图片中的文本内容。
    *   如果 LlamaParse 不可用或处理失败，则回退到标准的文件加载器。
*   **文档分块 (Chunk File)**:
    *   使用 `TokenTextSplitter` 对加载和解析后的文档进行分块。
    *   用户可以在界面上自定义 `块大小 (Chunk Size)` 和 `块重叠 (Chunk Overlap)` 参数。
*   **嵌入模型 (Embedding Model)**:
    *   使用本地的 `BAAI/bge-small-zh` 模型进行文本嵌入，无需额外 API 密钥。
*   **大语言模型 (LLM)**:
    *   使用 OpenAI 的 `gpt-4o` 模型作为生成答案的 LLM (需要 `OPENAI_API_KEY`)。
*   **向量索引与检索**:
    *   基于嵌入后的文本块构建向量索引。
    *   根据用户提问从索引中检索最相关的文本块。
*   **用户界面**:
    *   使用 Streamlit 构建交互式网页界面，方便用户上传文件、设置参数、提问并查看结果。

## 项目结构

```
.
├── rag_app.py          # RAG 核心逻辑 (加载, 解析, 分块, 索引, 查询引擎)
├── streamlit_app.py    # Streamlit 应用界面和流程控制
├── requirements.txt    # Python 依赖库
├── .env.example        # 环境变量模板文件
├── .gitignore          # 指定 Git忽略的文件和目录
└── README.md           # 本文档
```

## 安装与运行

### 1. 克隆仓库 (可选，如果您已在本地开发)

```bash
git clone https://github.com/173787247/RAG.git
cd RAG
```

### 2. 创建并激活虚拟环境 (推荐)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 文件并重命名为 `.env`：

```bash
# Windows
copy .env.example .env
# macOS/Linux
cp .env.example .env
```

然后，编辑 `.env` 文件，填入您的 API 密钥：

```env
LLAMA_CLOUD_API_KEY="YOUR_ACTUAL_LLAMA_CLOUD_API_KEY"
OPENAI_API_KEY="YOUR_ACTUAL_OPENAI_API_KEY"
```

*   `LLAMA_CLOUD_API_KEY`: 用于 LlamaParse 服务。如果留空，应用将使用标准文件加载器。
*   `OPENAI_API_KEY`: 用于 OpenAI LLM。如果留空，LLM 初始化会失败，查询功能将无法正常工作。

### 5. 运行 Streamlit 应用

```bash
streamlit run streamlit_app.py
```

应用启动后，通常可以在浏览器中打开 `http://localhost:8501` 进行访问。

## 使用说明

1.  **上传文档**: 在侧边栏通过文件上传器选择一个或多个支持的文档文件。
2.  **调整分块参数** (可选): 在侧边栏修改"块大小"和"块重叠"的默认值。
3.  **处理文档**: 点击"⚙️ 处理文档并构建索引"按钮。应用会加载、解析、分块文件，并构建向量索引。
4.  **提问**: 当查询引擎就绪后，在主界面输入您的问题，然后点击"提交查询"以获取答案。 