# 混合模型提供商使用指南

本指南介绍如何在LangGraph项目中配置和使用混合模型提供商功能，让不同的AI处理阶段使用不同的模型提供商。

## 功能概述

混合模型提供商功能允许您为以下三个AI处理阶段分别配置不同的模型提供商：

1. **查询生成阶段** (`query_generator`) - 生成搜索查询
2. **反思评估阶段** (`reflection`) - 评估研究结果的充分性
3. **答案生成阶段** (`answer`) - 生成最终答案

## 配置方法

### 1. 基础配置（单一提供商）

在 `.env` 文件中设置默认提供商：

```env
# 所有阶段都使用Gemini
MODEL_PROVIDER=gemini

# 或者所有阶段都使用硅基流动
MODEL_PROVIDER=siliconflow
```

### 2. 高级配置（混合提供商）

在 `.env` 文件中为特定阶段配置独立的提供商：

```env
# 默认提供商
MODEL_PROVIDER=gemini

# 为特定阶段配置不同的提供商
QUERY_GENERATOR_PROVIDER=gemini      # 查询生成使用Gemini
REFLECTION_PROVIDER=siliconflow      # 反思评估使用硅基流动
ANSWER_PROVIDER=gemini               # 答案生成使用Gemini
```

## 配置规则

1. **优先级**：特定阶段的配置优先于默认配置
2. **回退机制**：如果某个阶段没有配置特定提供商，将使用 `MODEL_PROVIDER` 的值
3. **空值处理**：留空或注释掉的配置项将使用默认提供商

## 使用场景示例

### 场景1：成本优化

```env
MODEL_PROVIDER=siliconflow           # 默认使用成本较低的硅基流动
ANSWER_PROVIDER=gemini               # 最终答案使用质量更高的Gemini
```

### 场景2：性能优化

```env
MODEL_PROVIDER=gemini                # 默认使用Gemini
QUERY_GENERATOR_PROVIDER=siliconflow # 查询生成使用速度更快的硅基流动
```

### 场景3：功能特化

```env
MODEL_PROVIDER=gemini
REFLECTION_PROVIDER=siliconflow      # 反思评估使用不同的模型获得多样化视角
```

## 环境变量完整列表

### 必需的API密钥

```env
# 如果使用Gemini（任何阶段）
GEMINI_API_KEY=your_gemini_api_key

# 如果使用硅基流动（任何阶段）
SILICONFLOW_API_KEY=your_siliconflow_api_key
```

### 提供商配置

```env
# 默认提供商（必需）
MODEL_PROVIDER=gemini|siliconflow

# 阶段特定提供商（可选）
QUERY_GENERATOR_PROVIDER=gemini|siliconflow
REFLECTION_PROVIDER=gemini|siliconflow
ANSWER_PROVIDER=gemini|siliconflow
```

## 验证配置

启动服务器时，系统会自动验证：

1. 所有使用的提供商都有对应的API密钥
2. 提供商名称是否有效（gemini 或 siliconflow）
3. 配置的一致性

如果配置有误，服务器启动时会显示详细的错误信息。

## 注意事项

1. **API密钥**：确保为所有使用的提供商配置了有效的API密钥
2. **成本控制**：不同提供商的定价策略不同，请根据使用量选择合适的组合
3. **性能差异**：不同模型的响应速度和质量可能有差异
4. **一致性**：混用可能导致不同阶段的输出风格略有差异

## 故障排除

### 常见错误

1. **API密钥未设置**
   ```
   ValueError: GEMINI_API_KEY is not set
   ```
   解决：在 `.env` 文件中添加对应的API密钥

2. **无效的提供商名称**
   ```
   ValueError: Unsupported model provider: invalid_provider
   ```
   解决：检查提供商名称是否为 `gemini` 或 `siliconflow`

3. **模块导入错误**
   ```
   ModuleNotFoundError: No module named 'agent.model_factory'
   ```
   解决：在backend目录下运行 `pip install -e .`

### 调试技巧

1. 检查 `.env` 文件的编码格式（应为UTF-8）
2. 确认环境变量名称拼写正确
3. 重启服务器以应用新的配置更改

## 更多信息

- [模型提供商配置文档](./model-providers.md)
- [API密钥获取指南](./api-keys.md)
- [性能优化建议](./performance-tuning.md)