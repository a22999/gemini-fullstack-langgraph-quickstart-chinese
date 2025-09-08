# =============================================================================
# LangGraph 共享模块 - 统一导出
# =============================================================================
# 本模块提供统一的模型工厂和配置管理功能，支持：
# 1. 混合模型工厂：统一的模型创建接口
# 2. 配置管理：灵活的配置系统
# 3. 向后兼容：支持 agent 和 chat 模块
# =============================================================================

# 导出配置类
from .configuration import ChatConfiguration, Configuration

# 导出模型工厂函数
from .model_factory import (
    create_mixed_chat_model,
    create_query_generator_model,
    create_reflection_model,
    create_answer_model,
    create_chat_model,
    create_model,  # Agent 兼容接口
    validate_mixed_model_config,
    get_mixed_model_provider,
    get_model_info,
    list_all_model_info,
    validate_config,  # 向后兼容别名
    get_model_provider,  # 向后兼容别名
)

__all__ = [
    # 配置类
    "ChatConfiguration",
    "Configuration",
    
    # 模型工厂函数
    "create_mixed_chat_model",
    "create_query_generator_model",
    "create_reflection_model",
    "create_answer_model",
    "create_chat_model",
    "create_model",
    
    # 配置和信息函数
    "validate_mixed_model_config",
    "get_mixed_model_provider",
    "get_model_info",
    "list_all_model_info",
    
    # 向后兼容别名
    "validate_config",
    "get_model_provider",
]