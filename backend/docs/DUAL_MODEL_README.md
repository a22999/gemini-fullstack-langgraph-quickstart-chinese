# 双模型聊天系统使用指南

## 概述

双模型聊天系统是一个创新的AI对话解决方案，它能够：

1. **并行查询**：同时向Gemini和硅基流动模型提出相同问题
2. **智能整合**：使用硅基流动模型将两个答案整合成更全面的回答
3. **透明展示**：显示每个模型的原始回答和最终整合结果

## 系统架构

```
用户问题
    ↓
并行查询节点
    ├── Gemini模型
    └── 硅基流动模型
    ↓
答案整合节点
    ↓
最终整合回答
```

## 环境配置

### 必需的API密钥

在使用双模型系统之前，您需要配置以下环境变量：

```bash
# Google Gemini API密钥
GEMINI_API_KEY=your_gemini_api_key_here

# 硅基流动API密钥
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
```

### 环境变量设置方法

#### 方法1：使用.env文件

在项目根目录创建`.env`文件：

```bash
GEMINI_API_KEY=your_gemini_api_key_here
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
```

#### 方法2：系统环境变量

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
$env:SILICONFLOW_API_KEY="your_siliconflow_api_key_here"
```

**Linux/macOS:**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export SILICONFLOW_API_KEY="your_siliconflow_api_key_here"
```

## 使用方法

### 1. 运行示例脚本

```bash
# 进入backend目录
cd backend

# 运行双模型示例
python src/chat/dual_model_example.py
```

### 2. 编程接口使用

```python
from src.chat.dual_model_graph import dual_model_graph
from langchain_core.messages import HumanMessage

# 创建初始状态
initial_state = {
    "messages": [HumanMessage(content="什么是人工智能？")],
    "processing_stage": "initial"
}

# 执行双模型图
result = dual_model_graph.invoke(initial_state)

# 获取结果
final_message = result["messages"][-1]
print("最终回答:", final_message.content)

# 查看详细信息
print("Gemini回答:", result.get("gemini_response"))
print("硅基流动回答:", result.get("siliconflow_response"))
print("整合回答:", result.get("integrated_response"))
```

### 3. 与LangGraph集成

```python
from src.chat.dual_model_graph import create_dual_model_graph
from src.chat.state import DualModelState

# 创建自定义图
graph = create_dual_model_graph()
compiled_graph = graph.compile()

# 使用自定义配置
config = {
    "configurable": {
        "thread_id": "conversation_1"
    }
}

result = compiled_graph.invoke(initial_state, config=config)
```

## 文件结构

```
src/chat/
├── dual_model_graph.py      # 双模型图定义
├── dual_model_example.py    # 使用示例
├── state.py                 # 状态定义（包含DualModelState）
├── model_factory.py         # 模型工厂
├── configuration.py         # 配置管理
└── DUAL_MODEL_README.md     # 本文档
```

## 核心组件说明

### DualModelState

扩展的状态类，包含：

- `messages`: 消息历史（继承自ChatState）
- `gemini_response`: Gemini模型的回答
- `siliconflow_response`: 硅基流动模型的回答
- `integrated_response`: 整合后的回答
- `processing_stage`: 处理阶段标识

### 处理节点

#### parallel_query_node
- 并行调用两个模型
- 使用ThreadPoolExecutor提高效率
- 包含完整的错误处理

#### integration_node
- 使用硅基流动模型整合答案
- 生成结构化的最终回答
- 保留原始回答的透明度

## 特性优势

### 1. 并行处理
- 同时调用两个模型，减少总响应时间
- 使用线程池确保高效执行

### 2. 智能整合
- 不是简单拼接，而是AI驱动的智能整合
- 识别冲突并提供判断
- 补充遗漏信息

### 3. 透明度
- 显示每个模型的原始回答
- 用户可以看到整合过程
- 便于比较和验证

### 4. 错误处理
- 单个模型失败不影响整体流程
- 详细的错误信息和恢复机制
- 配置验证确保环境正确

## 示例输出

```
🤖 双模型智能整合回答

人工智能（AI）是一门综合性学科，旨在创建能够模拟人类智能行为的计算机系统...

---
📊 模型回答详情

**Gemini模型回答：**
人工智能是指让机器具备类似人类智能的技术...

**硅基流动模型回答：**
人工智能是计算机科学的一个分支，专注于创建智能代理...
```

## 故障排除

### 常见问题

1. **API密钥错误**
   - 检查环境变量是否正确设置
   - 验证API密钥的有效性

2. **网络连接问题**
   - 确保网络连接正常
   - 检查防火墙设置

3. **模型调用失败**
   - 查看错误日志
   - 检查API配额和限制

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行双模型图
result = dual_model_graph.invoke(initial_state)
```

## 扩展和自定义

### 添加新的模型提供商

1. 在`model_factory.py`中添加新的创建函数
2. 更新`dual_model_graph.py`中的模型调用逻辑
3. 修改配置验证函数

### 自定义整合策略

修改`integration_node`函数中的整合提示，实现不同的整合策略：

```python
integration_prompt = f"""自定义整合提示..."""
```

### 添加更多处理节点

```python
# 添加预处理节点
graph.add_node("preprocess", preprocess_node)
graph.add_edge(START, "preprocess")
graph.add_edge("preprocess", "parallel_query")
```

## 性能优化

### 并发控制

```python
# 调整线程池大小
with ThreadPoolExecutor(max_workers=4) as executor:
    # 处理逻辑
```

### 缓存机制

```python
# 添加响应缓存
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_model_call(question):
    # 模型调用逻辑
```

## 许可证

本项目遵循MIT许可证。详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进双模型聊天系统！