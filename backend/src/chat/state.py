# =============================================================================
# LangGraph 简单聊天代理 - 状态管理模块
# =============================================================================
# 本文件定义了聊天代理的状态结构，用于在图的各个节点之间传递和管理数据
# 主要功能：
# 1. 定义聊天状态的数据结构
# 2. 管理消息历史和对话上下文
# 3. 支持状态的序列化和反序列化
# =============================================================================

from typing import Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# =============================================================================
# 聊天状态定义
# =============================================================================
class ChatState(TypedDict):
    """【聊天状态管理】
    定义聊天代理的核心状态结构，用于在图的各个节点之间传递数据。
    
    这个状态类比agent的OverallState更简单，专注于基本的聊天功能：
    - 消息历史管理
    - 对话上下文保持
    - 状态持久化支持
    
    Attributes:
        messages: 消息历史列表，使用add_messages进行自动合并和管理
                 支持用户消息、AI消息、系统消息等多种类型
    """
    
    # 消息历史列表
    # 使用Annotated和add_messages实现消息的自动合并和去重
    # add_messages函数会智能处理消息的添加、更新和排序
    messages: Annotated[List[BaseMessage], add_messages]


# =============================================================================
# 状态工具函数
# =============================================================================

def get_last_user_message(state: ChatState) -> str:
    """【获取最后一条用户消息】
    从状态中提取最后一条用户发送的消息内容。
    
    Args:
        state: 聊天状态对象
        
    Returns:
        str: 最后一条用户消息的文本内容，如果没有找到则返回空字符串
    """
    # 从后往前遍历消息列表，查找最后一条用户消息
    for message in reversed(state["messages"]):
        if hasattr(message, 'type') and message.type == "human":
            return message.content
    return ""


def get_conversation_history(state: ChatState, max_messages: int = 10) -> List[BaseMessage]:
    """【获取对话历史】
    获取指定数量的最近对话历史，用于上下文管理。
    
    Args:
        state: 聊天状态对象
        max_messages: 最大返回的消息数量，默认为10条
        
    Returns:
        List[BaseMessage]: 最近的对话历史消息列表
    """
    # 返回最近的max_messages条消息
    return state["messages"][-max_messages:] if len(state["messages"]) > max_messages else state["messages"]