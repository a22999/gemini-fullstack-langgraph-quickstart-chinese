# =============================================================================
# LangGraph 双模型聊天代理 - 图定义模块
# =============================================================================
# 本文件定义了双模型聊天代理图，实现：
# 1. 并行调用：同时询问Gemini和硅基流动模型
# 2. 答案收集：收集两个模型的回答
# 3. 智能整合：使用硅基流动模型整合两个答案
# 4. 对话历史：维护完整的对话上下文
# =============================================================================

# 系统模块
import os
from typing import Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain 核心模块
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# LangGraph 相关导入
from langgraph.graph import StateGraph, START, END

# 本地模块导入
from src.chat.state import ChatState, DualModelState, get_last_user_message
from src.chat.configuration import ChatConfiguration
from src.chat.model_factory import (
    create_gemini_model,
    create_siliconflow_model,
    validate_config
)


# =============================================================================
# 配置验证
# =============================================================================

def validate_dual_model_config():
    """【验证双模型配置】
    验证Gemini和硅基流动的API密钥是否都已配置。
    
    Raises:
        ValueError: 当任一API密钥未配置时抛出异常
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
    
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY 环境变量未设置，双模型模式需要Gemini API密钥")
    
    if not siliconflow_key:
        raise ValueError("SILICONFLOW_API_KEY 环境变量未设置，双模型模式需要硅基流动API密钥")


# =============================================================================
# 图节点定义
# =============================================================================

def parallel_query_node(state: DualModelState, config: RunnableConfig = None) -> Dict[str, Any]:
    """【并行查询节点】
    同时调用Gemini和硅基流动模型询问同一个问题。
    
    Args:
        state: 当前双模型聊天状态
        config: LangGraph运行时配置
        
    Returns:
        Dict[str, Any]: 包含两个模型回答的状态更新
    """
    # 验证配置
    try:
        validate_dual_model_config()
    except ValueError as e:
        return {
            "messages": [AIMessage(content=f"配置错误: {str(e)}")],
            "processing_stage": "error"
        }
    
    # 获取最后一条用户消息
    last_message = get_last_user_message(state)
    if not last_message:
        return {
            "messages": [AIMessage(content="抱歉，我没有收到您的消息。")],
            "processing_stage": "error"
        }
    
    # 获取聊天配置
    chat_config = ChatConfiguration.from_runnable_config(config)
    
    # 创建两个模型实例
    try:
        gemini_model = create_gemini_model(
            config=config,
            temperature=0.7,
            max_retries=2
        )
        
        siliconflow_model = create_siliconflow_model(
            config=config,
            temperature=0.7,
            max_retries=2
        )
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"模型创建失败: {str(e)}")],
            "processing_stage": "error"
        }
    
    # 准备查询消息
    query_messages = state["messages"]
    
    # 并行调用两个模型
    gemini_response = None
    siliconflow_response = None
    
    def query_gemini():
        try:
            response = gemini_model.invoke(query_messages)
            return "gemini", response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return "gemini", f"Gemini模型调用失败: {str(e)}"
    
    def query_siliconflow():
        try:
            response = siliconflow_model.invoke(query_messages)
            return "siliconflow", response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return "siliconflow", f"硅基流动模型调用失败: {str(e)}"
    
    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(query_gemini), executor.submit(query_siliconflow)]
        
        for future in as_completed(futures):
            try:
                model_name, response = future.result()
                if model_name == "gemini":
                    gemini_response = response
                elif model_name == "siliconflow":
                    siliconflow_response = response
            except Exception as e:
                # 如果某个模型完全失败，记录错误
                if gemini_response is None:
                    gemini_response = f"Gemini模型执行异常: {str(e)}"
                if siliconflow_response is None:
                    siliconflow_response = f"硅基流动模型执行异常: {str(e)}"
    
    return {
        "gemini_response": gemini_response,
        "siliconflow_response": siliconflow_response,
        "processing_stage": "parallel_query"
    }


def integration_node(state: DualModelState, config: RunnableConfig = None) -> Dict[str, Any]:
    """【答案整合节点】
    使用硅基流动模型整合Gemini和硅基流动的两个答案。
    
    Args:
        state: 当前双模型聊天状态
        config: LangGraph运行时配置
        
    Returns:
        Dict[str, Any]: 包含整合答案的状态更新
    """
    # 检查是否有两个回答
    if not state.get("gemini_response") or not state.get("siliconflow_response"):
        return {
            "messages": [AIMessage(content="无法获取到两个模型的回答，整合失败。")],
            "processing_stage": "error"
        }
    
    # 创建硅基流动模型用于整合
    try:
        integration_model = create_siliconflow_model(
            config=config,
            temperature=0.3,  # 降低温度以获得更稳定的整合结果
            max_retries=2
        )
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"整合模型创建失败: {str(e)}")],
            "processing_stage": "error"
        }
    
    # 获取原始问题
    last_message = get_last_user_message(state)
    original_question = last_message if isinstance(last_message, str) else (last_message.content if last_message else "未知问题")
    
    # 构建整合提示
    integration_prompt = f"""请你作为一个智能助手，整合以下两个AI模型对同一问题的回答，生成一个更全面、准确的综合回答。

    原始问题：{original_question}

    Gemini模型的回答：
    {state['gemini_response']}

    硅基流动模型的回答：
    {state['siliconflow_response']}

    请整合这两个回答，提供一个综合的、更完整的答案。要求：
    1. 保留两个回答中的优点和准确信息
    2. 如果有冲突，请指出并给出你的判断
    3. 补充任何遗漏的重要信息
    4. 确保回答逻辑清晰、结构完整
    5. 用中文回答

    综合回答："""
    
    # 调用整合模型
    try:
        integration_messages = [HumanMessage(content=integration_prompt)]
        response = integration_model.invoke(integration_messages)
        integrated_content = response.content if hasattr(response, 'content') else str(response)
        
        # 构建最终回答消息
        final_message = AIMessage(
            content=f"""🤖 **双模型智能整合回答**

            {integrated_content}

            ---
            📊 **模型回答详情**

            **Gemini模型回答：**
            {state['gemini_response']}

            **硅基流动模型回答：**
            {state['siliconflow_response']}"""
        )
        
        return {
            "messages": [final_message],
            "integrated_response": integrated_content,
            "processing_stage": "completed"
        }
        
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"答案整合失败: {str(e)}")],
            "processing_stage": "error"
        }


# =============================================================================
# 图构建与编译
# =============================================================================

def create_dual_model_graph() -> StateGraph:
    """【创建双模型聊天图】
    构建并返回支持双模型查询和整合的聊天处理状态图。
    
    Returns:
        StateGraph: 配置好的双模型聊天状态图
    """
    # 创建状态图
    graph = StateGraph(DualModelState)
    
    # 添加节点
    graph.add_node("parallel_query", parallel_query_node)
    graph.add_node("integration", integration_node)
    
    # 设置边
    graph.add_edge(START, "parallel_query")
    graph.add_edge("parallel_query", "integration")
    graph.add_edge("integration", END)
    
    return graph


# 编译图实例
dual_model_graph = create_dual_model_graph().compile()

# 默认导出
graph = dual_model_graph


# =============================================================================
# 图执行流程说明
# =============================================================================
"""
【双模型聊天图执行流程】

1. 开始 (START)
   ↓
2. 并行查询节点 (parallel_query)
   - 验证双模型配置
   - 同时调用Gemini和硅基流动模型
   - 收集两个模型的回答
   ↓
3. 整合节点 (integration)
   - 使用硅基流动模型整合两个答案
   - 生成综合回答
   - 展示详细的模型回答信息
   ↓
4. 结束 (END)
   - 返回包含整合答案的最终状态

【使用示例】
```python
from src.chat.dual_model_graph import dual_model_graph
from langchain_core.messages import HumanMessage

# 创建初始状态
initial_state = {
    "messages": [HumanMessage(content="什么是人工智能？")]
}

# 执行双模型图
result = dual_model_graph.invoke(initial_state)

# 获取整合后的回答
final_response = result["messages"][-1].content
print(final_response)
```

【特性说明】
- 并行处理：同时调用两个模型，提高效率
- 智能整合：使用AI模型整合答案，而非简单拼接
- 透明展示：显示每个模型的原始回答
- 错误处理：包含完整的异常处理机制
- 配置验证：确保双模型环境正确配置
"""