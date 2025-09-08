# Chat 混合模型与双模型聊天系统

这是一个完整的聊天系统实现，支持多种模型提供商和双模型并行处理。包含混合模型策略、双模型对比、演示脚本和测试工具等完整功能。

## 🚀 功能特性

### 核心功能
- **多提供商支持**: 同时支持 Gemini 和硅基流动
- **混合模型策略**: 不同处理阶段可使用不同模型
- **双模型并行**: 同时调用两个模型并整合答案
- **智能路由**: 根据配置自动选择合适的模型
- **向后兼容**: 保持与原有聊天功能的兼容性
- **配置灵活**: 支持环境变量和运行时配置

### 支持的处理阶段
- **chat**: 基础聊天对话
- **query_generator**: 查询生成
- **reflection**: 反思和优化
- **answer**: 最终答案生成

## 📁 目录结构与文件说明

```
src/chat/
├── __init__.py              # 包初始化文件
├── state.py                 # 聊天状态定义
├── configuration.py         # 混合模型配置管理
├── model_factory.py         # 模型工厂函数
├── graph.py                 # 基础聊天图定义
├── dual_model_graph.py      # 双模型聊天图定义
├── README.md                # 使用文档
├── examples/                # 示例文件夹
│   └── dual_model_example.py    # 双模型交互式示例
├── demos/                   # 演示文件夹
│   └── demo_dual_model.py       # 双模型演示脚本
└── tests/                   # 测试文件夹
    └── test_dual_model.py       # 双模型测试脚本
```

### 📄 文件详细说明

#### 核心模块文件

**`__init__.py`** - 包初始化文件
- 将chat目录标识为Python包
- 导入核心组件（graph对象）
- 定义包的公共API
- 控制可导入的对象

**`state.py`** - 聊天状态定义
- 定义基础聊天状态（ChatState）
- 定义双模型状态（DualModelState）
- 管理消息历史和对话上下文
- 提供状态工具函数（如获取最后用户消息）

**`configuration.py`** - 混合模型配置管理
- 定义ChatConfiguration配置类
- 支持多提供商配置（Gemini、硅基流动）
- 阶段特化配置（不同阶段使用不同模型）
- 环境变量集成和类型安全验证

**`model_factory.py`** - 模型工厂函数
- 实现混合模型工厂模式
- 支持多提供商模型创建
- 阶段特化模型选择
- 统一的模型创建接口
- 配置验证和模型信息查询

#### 图定义文件

**`graph.py`** - 基础聊天图定义
- 实现支持混合模型的聊天代理图
- 多提供商支持和智能对话
- 对话历史维护和灵活配置
- 运行时模型切换支持

**`dual_model_graph.py`** - 双模型聊天图定义
- 实现双模型并行调用架构
- 同时询问Gemini和硅基流动模型
- 答案收集和智能整合
- 完整的对话历史维护

#### 示例和演示文件

**`examples/dual_model_example.py`** - 双模型交互式示例
- 提供交互式双模型聊天体验
- 环境配置检查和验证
- 支持持续对话和用户输入
- 完整的错误处理和用户指导

**`demos/demo_dual_model.py`** - 双模型演示脚本
- 快速演示双模型功能
- 预设问题选择和自动化演示
- API密钥检查和环境配置
- 支持模拟模式（无API密钥时）

**`tests/test_dual_model.py`** - 双模型测试脚本
- 系统架构和功能测试
- 图结构和状态结构验证
- 支持无API密钥的基础测试
- 模拟API密钥的功能测试

#### 配置文件

**注意**：本模块使用项目根目录的 `backend/.env` 文件进行配置，无需单独的配置文件。

配置文件应包含：
- API 密钥配置（GOOGLE_API_KEY、SILICONFLOW_API_KEY）
- 模型提供商选择（MODEL_PROVIDER）
- 各阶段模型配置（可选）

## ⚙️ 配置说明

### 基础配置

1. **使用项目根目录配置**:
   配置文件位于 `backend/.env`，如果不存在请创建。

2. **设置 API 密钥**:
   ```bash
   # Gemini
   GEMINI_API_KEY=your_gemini_api_key
   
   # 硅基流动
   SILICONFLOW_API_KEY=your_siliconflow_api_key
   ```

3. **选择默认提供商**:
   ```bash
   MODEL_PROVIDER=gemini  # 或 siliconflow
   ```

### 混合模型配置

为不同阶段指定不同的模型提供商:

```bash
# 聊天使用 Gemini
CHAT_PROVIDER=gemini

# 查询生成使用硅基流动
QUERY_GENERATOR_PROVIDER=siliconflow

# 反思使用 Gemini
REFLECTION_PROVIDER=gemini

# 答案生成使用硅基流动
ANSWER_PROVIDER=siliconflow
```

## 🔧 使用方法

### 1. 基础聊天

```python
from src.chat.graph import graph
from src.chat.state import ChatState
from langchain_core.messages import HumanMessage

# 创建初始状态
state = ChatState(messages=[HumanMessage(content="你好！")])

# 执行聊天
result = graph.invoke(state)
print(result["messages"][-1].content)
```

### 2. 双模型并行聊天

```python
from src.chat.dual_model_graph import dual_model_graph
from src.chat.state import DualModelState
from langchain_core.messages import HumanMessage

# 创建双模型状态
state = {
    "messages": [HumanMessage(content="什么是人工智能？")],
    "gemini_response": None,
    "siliconflow_response": None,
    "integrated_response": None,
    "processing_stage": "initial"
}

# 执行双模型聊天
result = dual_model_graph.invoke(state)
print(result["integrated_response"])
```

### 3. 混合模型聊天

```python
from src.chat.graph import graph
from src.chat.state import ChatState
from langchain_core.messages import HumanMessage

# 创建初始状态
state = ChatState(messages=[HumanMessage(content="解释一下量子计算")])

# 使用混合模型配置
config = {
    "configurable": {
        "chat_provider": "gemini",
        "chat_model": "gemini-2.0-flash"
    }
}

# 执行聊天
result = graph.invoke(state, config=config)
print(result["messages"][-1].content)
```

### 4. 动态模型选择

```python
from src.chat.model_factory import create_mixed_chat_model

# 获取特定阶段的模型
chat_model = create_mixed_chat_model(stage="chat", temperature=0.8)
query_model = create_mixed_chat_model(stage="query_generator", temperature=0.3)

# 使用模型
response = chat_model.invoke([HumanMessage(content="你好")])
```

### 5. 模型信息查询

```python
from src.chat.model_factory import list_all_model_info

# 获取所有模型配置信息
model_info = list_all_model_info()
print(model_info)
```

## 🎮 快速开始

### 运行双模型演示

```bash
# 快速演示（自动选择问题）
echo 1 | python src/chat/demos/demo_dual_model.py

# 或者直接运行
python src/chat/demos/demo_dual_model.py
```

### 交互式双模型聊天

```bash
# 启动交互式聊天
python src/chat/examples/dual_model_example.py

# 或者通过管道输入问题
echo "什么是机器学习？" | python src/chat/examples/dual_model_example.py
```

### 系统测试

```bash
# 运行系统测试
python src/chat/tests/test_dual_model.py
```

## 🎯 使用场景

### 场景 1: 双模型对比分析
```bash
# 使用双模型系统获得更全面的答案
python src/chat/demos/demo_dual_model.py

# 适用于：
# - 重要决策需要多角度分析
# - 学术研究需要对比不同观点
# - 复杂问题需要综合性回答
```

### 场景 2: 成本优化
```bash
# 简单对话使用便宜的模型
CHAT_PROVIDER=siliconflow
CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct

# 复杂推理使用高级模型
ANSWER_PROVIDER=gemini
ANSWER_MODEL=gemini-2.0-flash
```

### 场景 3: 性能优化
```bash
# 快速响应阶段
CHAT_PROVIDER=gemini

# 深度分析阶段
REFLECTION_PROVIDER=siliconflow
REFLECTION_MODEL=Qwen/Qwen2.5-72B-Instruct
```

### 场景 4: 功能特化
```bash
# 中文对话优化
CHAT_PROVIDER=siliconflow
CHAT_MODEL=Qwen/Qwen2.5-72B-Instruct

# 英文生成优化
ANSWER_PROVIDER=gemini
ANSWER_MODEL=gemini-2.0-flash
```

## 🔍 API 参考

### 状态类

#### `ChatState`
- `messages`: 消息历史列表
- 支持自动消息合并和管理

#### `DualModelState`
- `messages`: 消息历史列表
- `gemini_response`: Gemini模型响应
- `siliconflow_response`: 硅基流动模型响应
- `integrated_response`: 整合后的最终响应
- `processing_stage`: 当前处理阶段

### 配置类

#### `ChatConfiguration`
- `from_runnable_config(config)`: 从运行时配置创建实例
- `get_model_for_stage(stage)`: 获取指定阶段的模型配置

### 模型工厂

#### `create_mixed_chat_model(stage, config, **kwargs)`
- `stage`: 处理阶段 ('chat', 'query_generator', 'reflection', 'answer')
- `config`: 运行时配置
- `**kwargs`: 模型参数 (temperature, max_tokens 等)

#### `validate_mixed_model_config()`
验证混合模型配置是否正确

#### `list_all_model_info(config=None)`
获取所有模型的配置信息

### 图对象

#### `graph` (基础聊天图)
- 支持混合模型的基础聊天功能
- 单模型响应模式

#### `dual_model_graph` (双模型聊天图)
- 并行调用Gemini和硅基流动模型
- 自动整合两个模型的响应

### 双模型函数

#### `validate_dual_model_config()`
验证双模型配置（需要两个API密钥）

#### `query_models_parallel(state)`
并行查询两个模型

#### `integrate_responses(state)`
整合两个模型的响应

## 🐛 故障排除

### 常见问题

#### 1. API密钥问题
```bash
# 错误：GEMINI_API_KEY 环境变量未设置
# 解决：在.env文件中设置API密钥
GEMINI_API_KEY=your_gemini_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
```

#### 2. 双模型配置问题
```bash
# 错误：双模型模式需要两个API密钥
# 解决：确保两个密钥都已设置
# 或者使用单模型模式
python src/chat/graph.py
```

#### 3. 模型调用失败
```bash
# 检查网络连接和API密钥有效性
# 运行测试脚本诊断问题
python src/chat/test_dual_model.py
```

#### 4. 环境配置问题
```bash
# 确保安装了所有依赖
pip install -r requirements.txt

# 检查Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 调试技巧

1. **使用测试脚本**：运行 `tests/test_dual_model.py` 检查系统状态
2. **检查日志**：观察控制台输出的详细错误信息
3. **模拟模式**：在没有API密钥时使用演示脚本的模拟模式
4. **逐步测试**：先测试单模型，再测试双模型功能

### 常见问题

1. **API 密钥错误**
   ```
   ValueError: GEMINI_API_KEY 环境变量未设置
   ```
   解决: 检查 `.env` 文件中的 API 密钥配置

2. **模型不存在**
   ```
   ValueError: 不支持的模型: invalid-model
   ```
   解决: 检查模型名称是否正确，参考配置示例

3. **提供商配置错误**
   ```
   ValueError: 不支持的模型提供商: invalid-provider
   ```
   解决: 确保提供商设置为 'gemini' 或 'siliconflow'

### 调试技巧

1. **启用调试模式**:
   ```bash
   DEBUG=true
   LOG_LEVEL=DEBUG
   ```

2. **检查模型配置**:
   ```python
   from src.chat.model_factory import list_all_model_info
   print(list_all_model_info())
   ```

3. **验证配置**:
   ```python
   from src.chat.model_factory import validate_mixed_model_config
   if not validate_mixed_model_config():
       print("配置验证失败")
   ```

## 📝 更新日志

### v1.0.0
- 初始版本
- 支持 Gemini 和硅基流动混合模型
- 完整的配置系统
- 向后兼容性支持

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个示例！

## 📄 许可证

本项目采用 MIT 许可证。