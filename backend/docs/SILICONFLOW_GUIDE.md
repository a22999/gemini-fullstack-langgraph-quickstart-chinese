# SiliconFlow 配置指南

本指南详细介绍如何在 LangGraph 智能研究代理中配置和使用硅基流动（SiliconFlow）作为 LLM 提供商。

## 什么是硅基流动？

硅基流动是一个提供多种开源大语言模型 API 服务的平台，支持包括 Qwen、DeepSeek、Llama 等在内的多种高质量模型。相比 Google Gemini，硅基流动提供了更多的模型选择和更灵活的定价方案。

## 配置步骤

### 1. 获取 API 密钥

1. 访问 [硅基流动官网](https://siliconflow.cn)
2. 注册账号并登录
3. 在控制台中创建 API 密钥
4. 复制您的 API 密钥

### 2. 环境配置

在 `backend/.env` 文件中添加以下配置：

```bash
# 设置模型提供商为硅基流动
MODEL_PROVIDER=siliconflow

# 硅基流动 API 密钥
SILICONFLOW_API_KEY=your_siliconflow_api_key_here

# 可选：配置特定模型（如不配置将使用默认模型）
QUERY_GENERATOR_MODEL=Qwen/Qwen2.5-72B-Instruct
REFLECTION_MODEL=deepseek-ai/DeepSeek-V2.5
ANSWER_MODEL=meta-llama/Meta-Llama-3.1-405B-Instruct
```

## 支持的模型

硅基流动支持多种高质量的开源模型，以下是推荐的模型配置：

### 推荐模型组合

#### 高性能配置（推荐）
```bash
QUERY_GENERATOR_MODEL=Qwen/Qwen2.5-72B-Instruct
REFLECTION_MODEL=deepseek-ai/DeepSeek-V2.5
ANSWER_MODEL=meta-llama/Meta-Llama-3.1-405B-Instruct
```

#### 平衡配置
```bash
QUERY_GENERATOR_MODEL=Qwen/Qwen2.5-32B-Instruct
REFLECTION_MODEL=Qwen/Qwen2.5-32B-Instruct
ANSWER_MODEL=Qwen/Qwen2.5-72B-Instruct
```

#### 经济配置
```bash
QUERY_GENERATOR_MODEL=Qwen/Qwen2.5-14B-Instruct
REFLECTION_MODEL=Qwen/Qwen2.5-14B-Instruct
ANSWER_MODEL=Qwen/Qwen2.5-32B-Instruct
```

### 可用模型列表

- **Qwen 系列**：
  - `Qwen/Qwen2.5-72B-Instruct` - 高性能通用模型
  - `Qwen/Qwen2.5-32B-Instruct` - 平衡性能和成本
  - `Qwen/Qwen2.5-14B-Instruct` - 轻量级高效模型

- **DeepSeek 系列**：
  - `deepseek-ai/DeepSeek-V2.5` - 优秀的推理能力
  - `deepseek-ai/DeepSeek-Coder-V2-Instruct` - 代码专用模型

- **Llama 系列**：
  - `meta-llama/Meta-Llama-3.1-405B-Instruct` - 超大规模模型
  - `meta-llama/Meta-Llama-3.1-70B-Instruct` - 高性能模型
  - `meta-llama/Meta-Llama-3.1-8B-Instruct` - 轻量级模型

## 混合配置示例

您也可以创建一个混合配置的示例，展示如何在 `backend/src/chat/` 目录中使用不同的模型提供商：

```python
# backend/src/chat/mixed_providers_example.py
from agent.model_factory import create_chat_model
from agent.configuration import Configuration

# 创建配置
config = Configuration()

# 使用 Gemini 进行查询生成
gemini_model = create_chat_model(
    model_name="gemini-2.0-flash-exp",
    provider="gemini",
    temperature=0.7
)

# 使用硅基流动进行反思和答案生成
siliconflow_model = create_chat_model(
    model_name="Qwen/Qwen2.5-72B-Instruct",
    provider="siliconflow",
    temperature=0.3
)
```

## 性能对比

| 特性 | Google Gemini | 硅基流动 |
|------|---------------|----------|
| 模型选择 | 有限（3-4个模型） | 丰富（20+模型） |
| 定价 | 按 token 计费 | 多种计费方式 |
| 响应速度 | 快 | 中等到快 |
| 中文支持 | 良好 | 优秀 |
| 开源模型 | 否 | 是 |
| API 稳定性 | 高 | 高 |

## 故障排除

### 常见问题

1. **API 密钥错误**
   ```
   错误：SILICONFLOW_API_KEY is not set
   解决：检查 .env 文件中的 API 密钥配置
   ```

2. **模型不存在**
   ```
   错误：Model not found
   解决：检查模型名称是否正确，参考上面的可用模型列表
   ```

3. **请求频率限制**
   ```
   错误：Rate limit exceeded
   解决：降低请求频率或升级 API 套餐
   ```

### 调试技巧

1. **启用详细日志**：
   ```bash
   export LOG_LEVEL=DEBUG
   ```

2. **测试 API 连接**：
   ```python
   from agent.model_factory import validate_model_config
   print(validate_model_config())
   ```

3. **检查模型响应**：
   ```python
   from agent.model_factory import create_chat_model
   
   model = create_chat_model(
       model_name="Qwen/Qwen2.5-14B-Instruct",
       provider="siliconflow"
   )
   
   response = model.invoke("Hello, how are you?")
   print(response.content)
   ```

## 最佳实践

1. **模型选择策略**：
   - 查询生成：使用较小的模型（如 Qwen2.5-14B）以提高速度
   - 反思分析：使用中等模型（如 Qwen2.5-32B）平衡性能和成本
   - 答案生成：使用大模型（如 Qwen2.5-72B 或 Llama-3.1-405B）确保质量

2. **成本优化**：
   - 根据任务复杂度选择合适的模型大小
   - 使用缓存减少重复请求
   - 监控 API 使用量

3. **性能优化**：
   - 调整 temperature 参数以平衡创造性和一致性
   - 使用并发请求提高吞吐量
   - 实施请求重试机制

## 支持与反馈

如果您在使用硅基流动时遇到问题，请：

1. 查看本指南的故障排除部分
2. 检查硅基流动官方文档
3. 在项目 GitHub 仓库中提交 Issue
4. 联系硅基流动技术支持

---

通过以上配置，您就可以在 LangGraph 智能研究代理中使用硅基流动的强大模型能力了！