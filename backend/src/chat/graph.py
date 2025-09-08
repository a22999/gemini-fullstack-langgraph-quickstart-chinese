# =============================================================================
# LangGraph 混合模型聊天代理 - 图定义模块
# =============================================================================
# 本文件定义了支持混合模型的聊天代理图，实现：
# 1. 多提供商支持：Gemini 和硅基流动
# 2. 智能对话：使用AI模型生成回复
# 3. 对话历史：维护完整的对话上下文
# 4. 灵活配置：支持运行时模型切换
# 5. 混合策略：不同场景使用不同模型
# =============================================================================

# 系统模块
import os
from typing import Any, Dict, Optional

# LangChain 核心模块
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# LangGraph 相关导入
from langgraph.graph import StateGraph, START, END

# 本地模块导入
from src.chat.state import ChatState, DualModelState, get_last_user_message
from src.shared.configuration import ChatConfiguration
from src.shared.model_factory import (
    create_mixed_chat_model,
    create_chat_model,
    create_gemini_model,
    create_siliconflow_model,
    validate_config as validate_mixed_model_config,
    get_model_info,
    list_all_model_info
)


# =============================================================================
# 配置管理和验证
# =============================================================================

def validate_chat_config():
    """【验证聊天配置】
    验证混合模型配置是否正确。
    
    Raises:
        ValueError: 当配置不正确时抛出异常
    """
    if not validate_mixed_model_config():
        # 获取详细的错误信息
        provider = os.getenv("MODEL_PROVIDER", "gemini").lower()
        if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY 环境变量未设置")
        elif provider == "siliconflow" and not os.getenv("SILICONFLOW_API_KEY"):
            raise ValueError("SILICONFLOW_API_KEY 环境变量未设置")
        else:
            raise ValueError(f"不支持的模型提供商或配置错误: {provider}")


def get_mixed_chat_model(
    config: Optional[RunnableConfig] = None,
    stage: str = "chat",
    temperature: float = 0.7,
    **kwargs
) -> Any:
    """【获取混合聊天模型】
    根据配置和阶段获取合适的AI模型实例。
    
    Args:
        config: LangGraph运行时配置
        stage: 处理阶段 ('chat', 'query_generator', 'reflection', 'answer')
        temperature: 模型温度参数
        **kwargs: 其他模型参数
        
    Returns:
        聊天模型实例（支持多种提供商）
    """
    return create_mixed_chat_model(
        stage=stage,
        config=config,
        temperature=temperature,
        **kwargs
    )


def get_chat_model_info(config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """【获取聊天模型信息】
    获取当前配置下的模型详细信息。
    
    Args:
        config: LangGraph运行时配置
        
    Returns:
        Dict: 模型配置信息
    """
    return list_all_model_info(config)


def get_chat_model(config: RunnableConfig = None):
    """【获取聊天模型（兼容性函数）】
    为了保持向后兼容性而保留的函数。
    
    Args:
        config: LangGraph运行时配置
        
    Returns:
        聊天模型实例
    """
    # 获取配置实例
    chat_config = ChatConfiguration.from_runnable_config(config)
    
    # 从配置中获取模型名称
    model_name = chat_config.chat_model
    if config and "configurable" in config:
        model_name = config["configurable"].get("chat_model", chat_config.chat_model)
    else:
        model_name = chat_config.chat_model
    
    # 使用模型工厂创建模型实例
    return create_chat_model(
        model_name=model_name,
        config=config,
        temperature=0.7,  # 设置适中的创造性
        max_retries=2,
    )


def mixed_chat(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """【混合模型聊天处理函数】
    使用混合模型策略处理用户输入并生成AI回复。
    
    Args:
        state: 当前聊天状态，包含消息历史
        config: LangGraph运行时配置
        
    Returns:
        Dict: 包含新消息的状态更新
    """
    # 验证配置
    try:
        validate_chat_config()
    except ValueError as e:
        return {"messages": [AIMessage(content=f"配置错误: {str(e)}")]}
    
    # 获取最后一条用户消息
    last_message = get_last_user_message(state)
    if not last_message:
        return {"messages": [AIMessage(content="抱歉，我没有收到您的消息。")]}
    
    # 获取混合聊天模型
    model = get_mixed_chat_model(config, stage="chat")
    
    # 生成回复
    try:
        # 使用模型生成回复
        response = model.invoke(state["messages"])
        
        # 返回新消息
        return {"messages": [response]}
        
    except Exception as e:
        # 错误处理
        error_message = f"生成回复时出错: {str(e)}"
        return {"messages": [AIMessage(content=error_message)]}


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
    
    # 验证配置
    try:
        validate_chat_config()
    except ValueError as e:
        return {"messages": [AIMessage(content=f"配置错误: {str(e)}")]}
    
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
# 双模型问答节点定义
# =============================================================================

def dual_model_initial_node(state: DualModelState, config: RunnableConfig = None) -> Dict[str, Any]:
    """【双模型初始化节点】
    初始化双模型问答流程，设置初始状态。
    
    Args:
        state: 双模型聊天状态
        config: LangGraph运行时配置
        
    Returns:
        Dict[str, Any]: 更新后的状态
    """
    # 验证配置
    try:
        validate_chat_config()
    except ValueError as e:
        return {
            "processing_stage": "error",
            "messages": [AIMessage(content=f"配置错误: {str(e)}")]
        }
    
    # 获取用户消息
    last_message = get_last_user_message(state)
    if not last_message:
        return {
            "processing_stage": "error",
            "messages": [AIMessage(content="抱歉，我没有收到您的消息。")]
        }
    
    return {
        "processing_stage": "parallel_query",
        "gemini_response": None,
        "siliconflow_response": None,
        "integrated_response": None
    }


def gemini_response_node(state: DualModelState, config: RunnableConfig = None) -> Dict[str, Any]:
    """【Gemini回答生成节点】
    使用Gemini模型生成回答。
    
    Args:
        state: 双模型聊天状态
        config: LangGraph运行时配置
        
    Returns:
        Dict[str, Any]: 包含Gemini回答的状态更新
    """
    try:
        # 创建Gemini模型，使用正确的模型名称
        gemini_model = create_mixed_chat_model(
            stage="chat",
            config=config,
            temperature=0.7,
            provider_override="gemini",
            model_override="gemini-1.5-flash"
        )
        
        # 生成回答
        response = gemini_model.invoke(state["messages"])
        gemini_content = response.content if hasattr(response, 'content') else str(response)
        
        return {"gemini_response": gemini_content}
        
    except Exception as e:
        return {"gemini_response": f"Gemini模型回答失败: {str(e)}"}


def siliconflow_response_node(state: DualModelState, config: RunnableConfig = None) -> Dict[str, Any]:
    """【硅基流动回答生成节点】
    使用硅基流动模型生成回答。
    
    Args:
        state: 双模型聊天状态
        config: LangGraph运行时配置
        
    Returns:
        Dict[str, Any]: 包含硅基流动回答的状态更新
    """
    try:
        # 创建硅基流动模型
        siliconflow_model = create_siliconflow_model(config=config, temperature=0.7)
        
        # 生成回答
        response = siliconflow_model.invoke(state["messages"])
        siliconflow_content = response.content if hasattr(response, 'content') else str(response)
        
        return {"siliconflow_response": siliconflow_content}
        
    except Exception as e:
        return {"siliconflow_response": f"硅基流动模型回答失败: {str(e)}"}


def integration_node(state: DualModelState, config: RunnableConfig = None) -> Dict[str, Any]:
    """【回答整合节点】
    整合两个模型的回答，生成最终的综合回答。
    
    Args:
        state: 双模型聊天状态
        config: LangGraph运行时配置
        
    Returns:
        Dict[str, Any]: 包含最终整合回答的状态更新
    """
    try:
        # 获取两个模型的回答
        gemini_resp = state.get("gemini_response", "无回答")
        siliconflow_resp = state.get("siliconflow_response", "无回答")
        
        # 构建整合提示
        integration_prompt = f"""请基于以下两个AI模型的回答，生成一个综合、准确、有用的最终回答：

**Gemini模型回答：**
{gemini_resp}

**硅基流动模型回答：**
{siliconflow_resp}

**要求：**
1. 综合两个回答的优点
2. 如果有冲突，选择更准确的信息
3. 保持回答的完整性和连贯性
4. 如果两个回答都有错误，请指出并提供正确信息
5. 最终回答应该简洁明了，直接回应用户的问题

**最终综合回答：**"""
        
        # 使用当前配置的模型进行整合（默认使用混合模型的chat阶段）
        integration_model = get_mixed_chat_model(config, stage="chat")
        
        # 创建整合消息
        integration_messages = [HumanMessage(content=integration_prompt)]
        
        # 生成整合回答
        response = integration_model.invoke(integration_messages)
        integrated_content = response.content if hasattr(response, 'content') else str(response)
        
        # 创建最终的AI消息
        final_message = AIMessage(content=integrated_content)
        
        return {
            "integrated_response": integrated_content,
            "processing_stage": "completed",
            "messages": [final_message]
        }
        
    except Exception as e:
        error_message = f"回答整合失败: {str(e)}"
        return {
            "integrated_response": error_message,
            "processing_stage": "error",
            "messages": [AIMessage(content=error_message)]
        }


def should_continue_to_integration(state: DualModelState) -> str:
    """【条件判断函数】
    判断是否可以进入整合阶段。
    
    Args:
        state: 双模型聊天状态
        
    Returns:
        str: 下一个节点名称
    """
    # 检查两个模型是否都已完成回答
    if (state.get("gemini_response") is not None and 
        state.get("siliconflow_response") is not None):
        return "integration"
    else:
        return "wait"  # 继续等待


# =============================================================================
# 图构建与编译
# =============================================================================

def create_mixed_chat_graph() -> StateGraph:
    """【创建混合模型聊天图】
    构建并返回支持混合模型的聊天处理状态图。
    
    Returns:
        StateGraph: 配置好的混合模型聊天状态图
    """
    # 创建状态图
    graph = StateGraph(ChatState)
    
    # 添加混合模型聊天节点
    graph.add_node("mixed_chat", mixed_chat)
    
    # 设置入口点
    graph.add_edge(START, "mixed_chat")
    
    # 设置出口点
    graph.add_edge("mixed_chat", END)
    
    return graph


def create_chat_graph() -> StateGraph:
    """【创建聊天图（兼容性函数）】
    构建并返回聊天处理的状态图。保持向后兼容性。
    
    Returns:
        StateGraph: 配置好的聊天状态图
    """
    # 创建状态图
    graph = StateGraph(ChatState)
    
    # 添加聊天节点
    graph.add_node("chat", chat_node)
    
    # 设置入口点
    graph.add_edge(START, "chat")
    
    # 设置出口点
    graph.add_edge("chat", END)
    
    return graph


def create_dual_model_chat_graph() -> StateGraph:
    """【创建双模型问答图】
    创建并编译双模型问答图实例，支持并行查询两个模型然后整合回答。
    
    Returns:
        StateGraph: 配置好的双模型问答状态图
    """
    # 创建状态图
    graph = StateGraph(DualModelState)
    
    # 添加节点
    graph.add_node("initial", dual_model_initial_node)
    graph.add_node("gemini_query", gemini_response_node)
    graph.add_node("siliconflow_query", siliconflow_response_node)
    graph.add_node("integration", integration_node)
    
    # 设置入口点
    graph.add_edge(START, "initial")
    
    # 添加边：从初始化节点到两个并行查询节点
    graph.add_edge("initial", "gemini_query")
    graph.add_edge("initial", "siliconflow_query")
    
    # 添加边：从两个查询节点到整合节点
    graph.add_edge("gemini_query", "integration")
    graph.add_edge("siliconflow_query", "integration")
    
    # 设置结束点
    graph.add_edge("integration", END)
    
    return graph


# 编译图实例
mixed_chat_graph = create_mixed_chat_graph().compile()
chat_graph = create_chat_graph().compile()
dual_model_chat_graph = create_dual_model_chat_graph().compile()

# 默认使用混合模型图
graph = mixed_chat_graph


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