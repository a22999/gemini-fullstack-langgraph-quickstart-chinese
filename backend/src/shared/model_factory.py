# =============================================================================
# LangGraph 共享模块 - 统一混合模型工厂模块
# =============================================================================
# 本文件实现了统一的混合模型工厂，支持：
# 1. 多提供商支持：Gemini 和硅基流动
# 2. 阶段特化：不同处理阶段使用不同的模型
# 3. 智能路由：根据配置自动选择合适的模型
# 4. 统一接口：提供一致的模型创建接口
# 5. 向后兼容：支持 agent 和 chat 模块的不同使用方式
# =============================================================================

# 系统模块
import os
from typing import Optional

# LangChain 核心模块
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel

# 模型提供商导入
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# 本地模块导入
from src.shared.configuration import ChatConfiguration, Configuration


# =============================================================================
# 混合模型工厂函数
# =============================================================================

def create_mixed_chat_model(
    stage: str,
    config: Optional[RunnableConfig] = None,
    temperature: float = 0.7,
    max_retries: int = 2,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> BaseChatModel:
    """
    【创建混合聊天模型】
    根据阶段和配置创建合适的聊天模型实例。
    
    Args:
        stage: 处理阶段 ('query_generator', 'reflection', 'answer', 'chat')
        config: LangGraph运行时配置
        temperature: 模型温度参数
        max_retries: 最大重试次数
        provider_override: 提供商覆盖（优先级最高）
        model_override: 模型名称覆盖（优先级最高）
        
    Returns:
        BaseChatModel: 聊天模型实例
        
    Raises:
        ValueError: 当提供商不支持或API密钥缺失时
    """
    # 获取配置实例
    chat_config = ChatConfiguration.from_runnable_config(config)
    
    # 确定使用的提供商和模型
    if provider_override and model_override:
        provider = provider_override
        model_name = model_override
    elif provider_override:
        provider = provider_override
        _, model_name = chat_config.get_model_for_stage(stage)
    else:
        provider, model_name = chat_config.get_model_for_stage(stage)
    
    # 如果有模型覆盖，使用覆盖的模型名称
    if model_override:
        model_name = model_override
    
    # 根据提供商创建模型
    if provider.lower() == "gemini":
        return _create_gemini_model(
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries
        )
    elif provider.lower() == "siliconflow":
        return _create_siliconflow_model(
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries
        )
    else:
        raise ValueError(f"不支持的模型提供商: {provider}")


def _create_gemini_model(
    model_name: str,
    temperature: float = 0.7,
    max_retries: int = 2
) -> ChatGoogleGenerativeAI:
    """
    【创建Gemini模型】
    创建Google Gemini聊天模型实例。
    
    Args:
        model_name: Gemini模型名称
        temperature: 模型温度参数
        max_retries: 最大重试次数
        
    Returns:
        ChatGoogleGenerativeAI: Gemini聊天模型实例
        
    Raises:
        ValueError: 当API密钥未设置时
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 环境变量未设置")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_retries=max_retries,
    )


def _create_siliconflow_model(
    model_name: str,
    temperature: float = 0.7,
    max_retries: int = 2
) -> ChatOpenAI:
    """
    【创建硅基流动模型】
    创建硅基流动聊天模型实例（使用OpenAI兼容接口）。
    
    Args:
        model_name: 硅基流动模型名称
        temperature: 模型温度参数
        max_retries: 最大重试次数
        
    Returns:
        ChatOpenAI: 硅基流动聊天模型实例
        
    Raises:
        ValueError: 当API密钥未设置时
    """
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("SILICONFLOW_API_KEY 环境变量未设置")
    
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        temperature=temperature,
        max_retries=max_retries,
    )


# =============================================================================
# 便捷函数
# =============================================================================

def create_query_generator_model(
    config: Optional[RunnableConfig] = None,
    **kwargs
) -> BaseChatModel:
    """
    【创建查询生成模型】
    创建用于查询生成的聊天模型。
    """
    return create_mixed_chat_model("query_generator", config, **kwargs)


def create_reflection_model(
    config: Optional[RunnableConfig] = None,
    **kwargs
) -> BaseChatModel:
    """
    【创建反思评估模型】
    创建用于反思评估的聊天模型。
    """
    return create_mixed_chat_model("reflection", config, **kwargs)


def create_answer_model(
    config: Optional[RunnableConfig] = None,
    **kwargs
) -> BaseChatModel:
    """
    【创建答案生成模型】
    创建用于答案生成的聊天模型。
    """
    return create_mixed_chat_model("answer", config, **kwargs)


def create_chat_model(
    config: Optional[RunnableConfig] = None,
    **kwargs
) -> BaseChatModel:
    """
    【创建聊天对话模型】
    创建用于聊天对话的聊天模型。
    """
    return create_mixed_chat_model("chat", config, **kwargs)


# =============================================================================
# 向后兼容性支持
# =============================================================================
# 为 agent 模块提供兼容的简化接口
# =============================================================================

def create_model(
    config: Optional[RunnableConfig] = None,
    temperature: float = 0.7,
    max_retries: int = 2,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseChatModel:
    """
    【创建模型 - Agent兼容接口】
    为 agent 模块提供向后兼容的模型创建接口。
    
    Args:
        config: LangGraph运行时配置
        temperature: 模型温度参数
        max_retries: 最大重试次数
        provider: 模型提供商（可选）
        model_name: 模型名称（可选）
        
    Returns:
        BaseChatModel: 聊天模型实例
    """
    # 使用 chat 阶段作为默认阶段
    return create_mixed_chat_model(
        stage="chat",
        config=config,
        temperature=temperature,
        max_retries=max_retries,
        provider_override=provider,
        model_override=model_name,
    )


# =============================================================================
# 配置验证函数
# =============================================================================

def validate_mixed_model_config() -> bool:
    """
    【验证混合模型配置】
    验证当前环境是否具备运行混合模型的条件。
    
    Returns:
        bool: 配置是否有效
    """
    # 检查基本提供商配置
    provider = os.getenv("MODEL_PROVIDER", "gemini").lower()
    
    # 收集所有可能使用的提供商
    providers_in_use = {provider}
    
    # 检查阶段特定提供商
    stage_providers = [
        os.getenv("QUERY_GENERATOR_PROVIDER", ""),
        os.getenv("REFLECTION_PROVIDER", ""),
        os.getenv("ANSWER_PROVIDER", ""),
        os.getenv("CHAT_PROVIDER", ""),
    ]
    
    for stage_provider in stage_providers:
        if stage_provider:
            providers_in_use.add(stage_provider.lower())
    
    # 验证每个使用的提供商都有对应的API密钥
    for provider_name in providers_in_use:
        if provider_name == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                return False
        elif provider_name == "siliconflow":
            if not os.getenv("SILICONFLOW_API_KEY"):
                return False
        elif provider_name:  # 非空但不支持的提供商
            return False
    
    return True


def get_mixed_model_provider(stage: str, config: Optional[RunnableConfig] = None) -> str:
    """
    【获取混合模型提供商】
    根据阶段和配置获取对应的模型提供商。
    
    Args:
        stage: 处理阶段名称
        config: LangGraph运行时配置
        
    Returns:
        str: 提供商名称
    """
    chat_config = ChatConfiguration.from_runnable_config(config)
    provider, _ = chat_config.get_model_for_stage(stage)
    return provider


# =============================================================================
# 模型信息函数
# =============================================================================

def get_model_info(stage: str, config: Optional[RunnableConfig] = None) -> dict:
    """
    【获取模型信息】
    获取指定阶段的详细模型信息。
    
    Args:
        stage: 处理阶段名称
        config: LangGraph运行时配置
        
    Returns:
        dict: 模型信息字典
    """
    chat_config = ChatConfiguration.from_runnable_config(config)
    provider, model_name = chat_config.get_model_for_stage(stage)
    
    return {
        "stage": stage,
        "provider": provider,
        "model_name": model_name,
        "api_key_set": bool(os.getenv(f"{provider.upper()}_API_KEY")),
    }


def list_all_model_info(config: Optional[RunnableConfig] = None) -> dict:
    """
    【列出所有模型信息】
    获取所有阶段的模型配置信息。
    
    Args:
        config: LangGraph运行时配置
        
    Returns:
        dict: 所有阶段的模型信息
    """
    stages = ["query_generator", "reflection", "answer", "chat"]
    
    return {
        stage: get_model_info(stage, config)
        for stage in stages
    }


# =============================================================================
# 向后兼容性别名和函数
# =============================================================================
# 为保持与现有代码的兼容性，提供一些常用的别名和函数
# =============================================================================

# Agent 模块兼容别名
get_model_provider = get_mixed_model_provider
validate_config = validate_mixed_model_config

# Chat 模块兼容函数
def create_gemini_model(
    config: Optional[RunnableConfig] = None,
    temperature: float = 0.7,
    max_retries: int = 2,
    **kwargs
) -> BaseChatModel:
    """创建 Gemini 模型 - 向后兼容函数"""
    return create_mixed_chat_model(
        stage="chat",
        config=config,
        temperature=temperature,
        max_retries=max_retries,
        provider_override="gemini",
        **kwargs
    )

def create_siliconflow_model(
    config: Optional[RunnableConfig] = None,
    temperature: float = 0.7,
    max_retries: int = 2,
    **kwargs
) -> BaseChatModel:
    """创建硅基流动模型 - 向后兼容函数"""
    return create_mixed_chat_model(
        stage="chat",
        config=config,
        temperature=temperature,
        max_retries=max_retries,
        provider_override="siliconflow",
        **kwargs
    )