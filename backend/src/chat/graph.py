# =============================================================================
# LangGraph 简单聊天代理 - 图定义模块
# =============================================================================
# 本文件定义了一个简单的聊天代理图，实现基本的对话功能
# 主要功能：
# 1. 接收用户消息
# 2. 使用AI模型生成回复
# 3. 维护对话历史
# 4. 支持多轮对话
# =============================================================================

# 系统模块
import os
from typing import Any, Dict

# LangChain 核心模块
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph 图构建模块
from langgraph.graph import StateGraph, START, END

# 本地模块
from src.chat.state import ChatState, get_last_user_message


# =============================================================================
# 配置管理
# =============================================================================
def get_chat_model(config: RunnableConfig = None) -> ChatGoogleGenerativeAI:
    """【获取聊天模型】
    根据配置获取用于聊天的AI模型实例。
    
    Args:
        config: LangGraph运行时配置，可包含模型设置
        
    Returns:
        ChatGoogleGenerativeAI: 配置好的聊天模型实例
    """
    # 从配置中获取模型名称，默认使用gemini-2.0-flash
    model_name = "gemini-2.0-flash"
    if config and "configurable" in config:
        model_name = config["configurable"].get("chat_model", model_name)
    
    # 创建并返回模型实例
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7,  # 设置适中的创造性
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


# =============================================================================
# 图节点定义
# =============================================================================

def chat_node(state: ChatState, config: RunnableConfig = None) -> Dict[str, Any]:
    """【聊天节点】
    处理用户消息并生成AI回复的核心节点。
    
    功能说明：
    1. 获取用户的最新消息
    2. 调用AI模型生成回复
    3. 将AI回复添加到消息历史中
    4. 返回更新后的状态
    
    Args:
        state: 当前聊天状态，包含消息历史
        config: LangGraph运行时配置
        
    Returns:
        Dict[str, Any]: 包含新AI消息的状态更新
    """
    
    # 获取聊天模型实例
    model = get_chat_model(config)
    
    # 获取当前的消息历史
    messages = state["messages"]
    
    # 调用模型生成回复
    # 模型会基于完整的对话历史生成上下文相关的回复
    try:
        response = model.invoke(messages)
        
        # 确保响应是AIMessage类型
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))
            
    except Exception as e:
        # 错误处理：如果模型调用失败，返回错误消息
        response = AIMessage(
            content=f"抱歉，我遇到了一个错误：{str(e)}。请稍后再试。"
        )
    
    # 返回包含新消息的状态更新
    return {
        "messages": [response]
    }


# =============================================================================
# 图构建与编译
# =============================================================================

# 创建状态图实例
# 使用ChatState作为状态类型，定义图的数据流结构
graph_builder = StateGraph(ChatState)

# 添加聊天节点
# "chat"是节点的唯一标识符，chat_node是实际的处理函数
graph_builder.add_node("chat", chat_node)

# 设置图的边连接
# 简单的线性流程：START -> chat -> END
graph_builder.add_edge(START, "chat")
graph_builder.add_edge("chat", END)

# 编译图
# 将图构建器编译成可执行的图对象
graph = graph_builder.compile()


# =============================================================================
# 图执行流程说明
# =============================================================================
"""
【聊天图执行流程】

1. 开始 (START)
   ↓
2. 聊天节点 (chat)
   - 接收包含用户消息的状态
   - 调用AI模型生成回复
   - 将AI回复添加到消息历史
   ↓
3. 结束 (END)
   - 返回包含完整对话历史的最终状态

【使用示例】
```python
# 创建初始状态
initial_state = {
    "messages": [HumanMessage(content="你好！")]
}

# 执行图
result = graph.invoke(initial_state)

# 获取AI回复
ai_response = result["messages"][-1].content
print(ai_response)
```

【特性说明】
- 简单直接：单节点处理，适合基本聊天场景
- 状态管理：自动维护完整的对话历史
- 错误处理：包含基本的异常处理机制
- 可配置：支持通过config调整模型参数
- 可扩展：可以轻松添加更多节点实现复杂功能
"""