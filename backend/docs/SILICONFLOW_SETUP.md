# 硅基流动 API 配置指南

本文档介绍如何在 LangGraph 项目中配置和使用硅基流动 API。

## 概述

硅基流动提供了与 OpenAI 兼容的 API 接口，支持多种开源大语言模型。本项目现在支持在 Google Gemini 和硅基流动之间切换。

## 配置步骤

### 1. 获取硅基流动 API 密钥

1. 访问 [硅基流动官网](https://siliconflow.cn/)
2. 注册账号并登录
3. 在控制台中创建 API 密钥
4. 复制您的 API 密钥

### 2. 配置环境变量

在 `backend/.env` 文件中添加以下配置：

```bash
# 硅基流动 API 配置
SILICONFLOW_API_KEY=your_actual_api_key_here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# 设置模型提供商为硅基流动
MODEL_PROVIDER=siliconflow

# 硅基流动模型配置
QUERY_GENERATOR_MODEL=Qwen/Qwen2.5-7B-Instruct
REFLECTION_MODEL=Qwen/Qwen2.5-14B-Instruct
ANSWER_MODEL=Qwen/Qwen2.5-32B-Instruct
```

### 3. 可用模型列表

硅基流动支持多种开源模型，以下是推荐的配置：

#### 轻量级配置（快速响应）
```bash
QUERY_GENERATOR_MODEL=Qwen/Qwen2.5-7B-Instruct
REFLECTION_MODEL=Qwen/Qwen2.5-7B-Instruct
ANSWER_MODEL=Qwen/Qwen2.5-14B-Instruct
```

#### 平衡配置（推荐）
```bash
QUERY_GENERATOR_MODEL=Qwen/Qwen2.5-7B-Instruct
REFLECTION_MODEL=Qwen/Qwen2.5-14B-Instruct
ANSWER_MODEL=Qwen/Qwen2.5-32B-Instruct
```

#### 高性能配置（最佳质量）
```bash
QUERY_GENERATOR_MODEL=Qwen/Qwen2.5-14B-Instruct
REFLECTION_MODEL=Qwen/Qwen2.5-32B-Instruct
ANSWER_MODEL=Qwen/Qwen2.5-72B-Instruct
```

#### 其他支持的模型
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`
- `deepseek-ai/DeepSeek-V2.5`
- `01-ai/Yi-1.5-34B-Chat`
- `THUDM/glm-4-9b-chat`

## 使用方法

### 切换到硅基流动

1. 修改 `.env` 文件中的 `MODEL_PROVIDER`：
   ```bash
   MODEL_PROVIDER=siliconflow
   ```

2. 重启 LangGraph 服务器：
   ```bash
   langgraph dev
   ```

### 切换回 Gemini

1. 修改 `.env` 文件中的 `MODEL_PROVIDER`：
   ```bash
   MODEL_PROVIDER=gemini
   ```

2. 重启 LangGraph 服务器

## 注意事项

### 1. Google Search 功能

**重要**：即使使用硅基流动作为主要模型提供商，Google Search 功能仍然需要 Gemini API 密钥，因为它使用 Google 的搜索服务。

确保 `.env` 文件中仍然配置了：
```bash
GEMINI_API_KEY=your_gemini_api_key
```

### 2. 模型兼容性

- 硅基流动使用 OpenAI 兼容的 API 格式
- 所有 LangChain 的 OpenAI 功能都可以正常使用
- 结构化输出（structured output）功能完全支持

### 3. 成本考虑

- 硅基流动的定价通常比 OpenAI 更具竞争力
- 不同模型的成本差异较大，建议根据需求选择合适的模型
- 可以为不同阶段使用不同规模的模型来优化成本

## 故障排除

### 1. API 密钥错误

如果遇到认证错误，请检查：
- API 密钥是否正确
- API 密钥是否有足够的配额
- 网络连接是否正常

### 2. 模型不可用

如果某个模型不可用，请：
- 检查模型名称是否正确
- 确认该模型在硅基流动平台上可用
- 尝试使用其他模型

### 3. 性能问题

如果响应较慢：
- 尝试使用更小的模型
- 检查网络延迟
- 考虑调整 `max_retries` 参数

## 示例配置文件

完整的 `.env` 配置示例：

```bash
# API 密钥
GEMINI_API_KEY=your_gemini_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# 模型提供商选择
MODEL_PROVIDER=siliconflow

# 硅基流动模型配置
QUERY_GENERATOR_MODEL=Qwen/Qwen2.5-7B-Instruct
REFLECTION_MODEL=Qwen/Qwen2.5-14B-Instruct
ANSWER_MODEL=Qwen/Qwen2.5-32B-Instruct

# 流程控制配置
NUMBER_OF_INITIAL_QUERIES=5
MAX_RESEARCH_LOOPS=3
```

## 支持

如果您在配置过程中遇到问题，请：

1. 检查本文档的故障排除部分
2. 查看 LangGraph 的日志输出
3. 确认所有依赖项已正确安装
4. 验证网络连接和 API 访问权限