# Gemini 全栈 LangGraph 快速入门

本项目展示了一个使用 React 前端和 LangGraph 驱动的后端智能体的全栈应用程序。该智能体旨在通过动态生成搜索词、使用 Google 搜索查询网络、反思结果以识别知识缺口，并迭代优化搜索，直到能够提供有充分支持和引用的答案，从而对用户查询进行全面研究。该应用程序作为使用 LangGraph 和 Google Gemini 模型构建研究增强型对话 AI 的示例。

<img src="./app.png" title="Gemini Fullstack LangGraph" alt="Gemini Fullstack LangGraph" width="90%">

## 功能特性

- 💬 使用 React 前端和 LangGraph 后端的全栈应用程序
- 🧠 由 LangGraph 智能体驱动的高级研究和对话 AI
- 🔍 使用 Google Gemini 模型动态生成搜索查询
- 🌐 通过 Google Search API 集成网络研究
- 🤔 反思推理以识别知识缺口并优化搜索
- 📄 从收集的来源生成带引用的答案
- 🔄 开发期间前端和后端的热重载

## 项目结构

项目分为两个主要目录：

-   `frontend/`：包含使用 Vite 构建的 React 应用程序
-   `backend/`：包含 LangGraph/FastAPI 应用程序，包括研究智能体逻辑

## 开始使用：开发和本地测试

按照以下步骤在本地运行应用程序进行开发和测试。

**1. 前置条件：**

-   Node.js 和 npm（或 yarn/pnpm）
-   Python 3.11+
-   **`GEMINI_API_KEY`**：后端智能体需要 Google Gemini API 密钥
    1.  导航到 `backend/` 目录
    2.  通过复制 `backend/.env.example` 文件创建名为 `.env` 的文件
    3.  打开 `.env` 文件并添加您的 Gemini API 密钥：`GEMINI_API_KEY="YOUR_ACTUAL_API_KEY"`

**2. 安装依赖：**

**后端：**

```bash
cd backend
pip install .
```

**前端：**

```bash
cd frontend
npm install
```

**3. 运行开发服务器：**

**后端和前端：**

```bash
make dev
```
这将运行后端和前端开发服务器。在浏览器中打开并导航到前端开发服务器 URL（例如，`http://localhost:5173/app`）。

_或者，您可以分别运行后端和前端开发服务器。对于后端，在 `backend/` 目录中打开终端并运行 `langgraph dev`。后端 API 将在 `http://127.0.0.1:2024` 可用。它还会打开一个浏览器窗口到 LangGraph UI。对于前端，在 `frontend/` 目录中打开终端并运行 `npm run dev`。前端将在 `http://localhost:5173` 可用。_

## 后端智能体工作原理（高级概述）

后端的核心是在 `backend/src/agent/graph.py` 中定义的 LangGraph 智能体。它遵循以下步骤：

<img src="./agent.png" title="Agent Flow" alt="Agent Flow" width="50%">

1.  **生成初始查询：** 基于您的输入，使用 Gemini 模型生成一组初始搜索查询
2.  **网络研究：** 对于每个查询，使用 Gemini 模型和 Google Search API 查找相关网页
3.  **反思和知识缺口分析：** 智能体分析搜索结果以确定信息是否充分或是否存在知识缺口。它使用 Gemini 模型进行此反思过程
4.  **迭代优化：** 如果发现缺口或信息不足，它会生成后续查询并重复网络研究和反思步骤（最多配置的最大循环次数）
5.  **最终答案：** 一旦研究被认为充分，智能体使用 Gemini 模型将收集的信息合成为连贯的答案，包括来自网络来源的引用

## CLI 示例

对于快速的一次性问题，您可以从命令行执行智能体。脚本 `backend/examples/cli_research.py` 运行 LangGraph 智能体并打印最终答案：

```bash
cd backend
python examples/cli_research.py "可再生能源的最新趋势是什么？"
```

## 部署

在生产环境中，后端服务器提供优化的静态前端构建。LangGraph 需要 Redis 实例和 Postgres 数据库。Redis 用作发布-订阅代理，以启用来自后台运行的流式实时输出。Postgres 用于存储助手、线程、运行、持久化线程状态和长期记忆，以及管理具有"恰好一次"语义的后台任务队列状态。有关如何部署后端服务器的更多详细信息，请查看 [LangGraph 文档](https://langchain-ai.github.io/langgraph/concepts/deployment_options/)。以下是如何构建包含优化前端构建和后端服务器的 Docker 镜像并通过 `docker-compose` 运行它的示例。

_注意：对于 docker-compose.yml 示例，您需要 LangSmith API 密钥，您可以从 [LangSmith](https://smith.langchain.com/settings) 获取。_

_注意：如果您没有运行 docker-compose.yml 示例或将后端服务器暴露到公共互联网，您应该更新 `frontend/src/App.tsx` 文件中的 `apiUrl` 为您的主机。目前 `apiUrl` 设置为 docker-compose 的 `http://localhost:8123` 或开发的 `http://localhost:2024`。_

**1. 构建 Docker 镜像：**

   从**项目根目录**运行以下命令：
   ```bash
   docker build -t gemini-fullstack-langgraph -f Dockerfile .
   ```
**2. 运行生产服务器：**

   ```bash
   GEMINI_API_KEY=<your_gemini_api_key> LANGSMITH_API_KEY=<your_langsmith_api_key> docker-compose up
   ```

在浏览器中打开并导航到 `http://localhost:8123/app/` 查看应用程序。API 将在 `http://localhost:8123` 可用。

## 使用的技术

- [React](https://reactjs.org/)（使用 [Vite](https://vitejs.dev/)）- 用于前端用户界面
- [Tailwind CSS](https://tailwindcss.com/) - 用于样式设计
- [Shadcn UI](https://ui.shadcn.com/) - 用于组件
- [LangGraph](https://github.com/langchain-ai/langgraph) - 用于构建后端研究智能体
- [Google Gemini](https://ai.google.dev/models/gemini) - 用于查询生成、反思和答案合成的 LLM

## 许可证

本项目采用 Apache License 2.0 许可证。详情请参阅 [LICENSE](LICENSE) 文件。