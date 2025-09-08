# =============================================================================
# LangGraph 智能研究代理 - 模型工厂模块（共享版本）
# =============================================================================
# 本文件现在使用共享的模型工厂模块，提供：
# 1. 向后兼容的接口
# 2. 统一的模型创建功能
# 3. 简化的配置管理
# 4. 与 chat 模块的一致性
# =============================================================================

import os
from typing import Any, Dict, Optional, Union
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# 使用共享的模型工厂和配置
from src.shared import (
    create_model,
    Configuration,
    validate_config,
    get_model_provider,
    get_model_info as shared_get_model_info,
)


def create_chat_model(
    model_name: str,
    config: Optional[RunnableConfig] = None,
    temperature: float = 0.7,
    max_retries: int = 2,
    **kwargs: Any
) -> Union[ChatGoogleGenerativeAI, ChatOpenAI]:
    """
    【创建聊天模型 - 向后兼容接口】
    根据配置创建聊天模型实例，现在使用共享的模型工厂。
    
    Args:
        model_name: 模型名称
        config: LangGraph 运行时配置
        temperature: 模型温度参数 (0.0-1.0)
        max_retries: 最大重试次数
        **kwargs: 其他模型参数
        
    Returns:
        聊天模型实例 (Gemini 或硅基流动)
        
    Raises:
        ValueError: 当配置错误时
    """
    # 使用共享的模型工厂创建模型
    return create_model(
        config=config,
        temperature=temperature,
        max_retries=max_retries,
        model_name=model_name,
        **kwargs
    )


# =============================================================================
# 注意：私有函数已移至共享模块
# =============================================================================
# _create_gemini_model 和 _create_siliconflow_model 函数现在位于
# src.shared.model_factory 模块中，通过 create_model 函数统一调用。
# =============================================================================





def get_model_info(config: Optional[RunnableConfig] = None) -> Dict[str, str]:
    """
    【获取模型信息 - 向后兼容接口】
    返回当前配置的模型信息，现在使用共享的模型信息功能。
    
    Args:
        config: LangGraph 运行时配置
        
    Returns:
        包含模型信息的字典
    """
    # 使用共享模块获取 chat 阶段的模型信息
    shared_info = shared_get_model_info("chat", config)
    
    # 转换为 agent 模块期望的格式
    configurable = Configuration.from_runnable_config(config)
    
    info = {
        "model_provider": shared_info["provider"],
        "query_generator_model": configurable.query_generator_model,
        "reflection_model": configurable.reflection_model,
        "answer_model": configurable.answer_model,
    }
    
    # 添加提供商特定信息
    if shared_info["provider"].lower() == "gemini":
        info["api_endpoint"] = "Google Gemini API"
        info["api_key_env"] = "GEMINI_API_KEY"
    elif shared_info["provider"].lower() == "siliconflow":
        info["api_endpoint"] = "https://api.siliconflow.cn/v1"
        info["api_key_env"] = "SILICONFLOW_API_KEY"
    
    return info


def validate_model_config(config: Optional[RunnableConfig] = None) -> bool:
    """
    【验证模型配置 - 向后兼容接口】
    检查当前模型配置是否有效，现在使用共享的验证功能。
    
    Args:
        config: LangGraph 运行时配置
        
    Returns:
        配置是否有效
    """
    # 使用共享模块的配置验证功能
    return validate_config()


# =============================================================================
# 向后兼容性别名
# =============================================================================
# 为保持与现有 agent 代码的完全兼容性
# =============================================================================

# 提供商获取函数别名
get_model_provider = get_model_provider

# 配置验证函数别名
validate_config_alias = validate_config