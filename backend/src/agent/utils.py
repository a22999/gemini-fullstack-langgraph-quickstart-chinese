# -*- coding: utf-8 -*-
"""
工具函数模块 (Utils Module)

本文件包含了LangGraph研究代理所需的各种工具函数，主要功能包括：
1. 研究主题提取 - 从消息历史中提取研究主题
2. URL解析和缩短 - 将长URL转换为短URL映射
3. 引用标记插入 - 在文本中插入引用标记
4. 引用信息提取 - 从Gemini模型响应中提取引用信息

这些工具函数支持整个研究流程中的文本处理、URL管理和引用格式化等核心功能。
"""

from typing import Any, Dict, List  # 类型提示：任意类型、字典、列表
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage  # LangChain消息类型


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    从消息列表中提取研究主题 (Extract Research Topic from Messages)
    
    功能说明：
    - 如果只有一条消息，直接返回该消息内容作为研究主题
    - 如果有多条消息（对话历史），将所有消息组合成一个包含角色标识的字符串
    
    参数：
        messages (List[AnyMessage]): 消息列表，包含用户和助手的对话历史
    
    返回：
        str: 提取的研究主题字符串
    """
    # 检查请求是否有历史记录，并将消息组合成单个字符串
    if len(messages) == 1:
        # 单条消息：直接使用最后一条消息的内容
        research_topic = messages[-1].content
    else:
        # 多条消息：组合所有消息并添加角色标识
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    URL解析和缩短映射 (URL Resolution and Shortening Mapping)
    
    功能说明：
    - 将Vertex AI搜索返回的长URL转换为带有唯一标识符的短URL
    - 确保每个原始URL都获得一致的缩短形式，同时保持唯一性
    - 用于在引用中显示更简洁的URL格式
    
    参数：
        urls_to_resolve (List[Any]): 需要解析的URL对象列表
        id (int): 用于生成唯一短URL的标识符
    
    返回：
        Dict[str, str]: 原始URL到短URL的映射字典
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/"  # 短URL前缀
    urls = [site.web.uri for site in urls_to_resolve]  # 提取所有URL

    # 创建一个字典，将每个唯一URL映射到其首次出现的索引
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            # 为每个唯一URL生成短URL：前缀 + ID + 索引
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def insert_citation_markers(text, citations_list):
    """
    在文本中插入引用标记 (Insert Citation Markers into Text)
    
    功能说明：
    - 根据起始和结束索引在文本字符串中插入引用标记
    - 采用从后向前的插入策略，避免索引位置偏移问题
    - 将引用信息格式化为Markdown链接格式
    
    参数：
        text (str): 原始文本字符串
        citations_list (list): 引用信息列表，每个字典包含：
                              - 'start_index': 引用开始位置
                              - 'end_index': 引用结束位置  
                              - 'segments': 引用片段信息
                              索引基于原始文本计算
    
    返回：
        str: 插入引用标记后的文本
    """
    # 按end_index降序排序引用列表
    # 如果end_index相同，则按start_index降序排序
    # 这确保从字符串末尾开始插入，不会影响前面部分的索引位置
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        # 这些索引指向*原始*文本中的位置
        # 由于我们从末尾开始迭代，相对于已处理的字符串部分，索引仍然有效
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        # 构建引用标记：格式化为Markdown链接
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        # 在原始end_idx位置插入引用标记
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text


def get_citations(response, resolved_urls_map):
    """
    从Gemini模型响应中提取和格式化引用信息 (Extract and Format Citation Information from Gemini Response)
    
    功能说明：
    - 处理响应中提供的grounding元数据，构建引用对象列表
    - 每个引用对象包含文本片段的起始和结束索引
    - 包含格式化的Markdown链接，指向支持的网络块
    - 与URL解析映射配合使用，提供简洁的引用格式
    
    参数：
        response: Gemini模型的响应对象，预期包含
                 `candidates[0].grounding_metadata`结构
        resolved_urls_map: URL映射字典，将块URI映射到解析的URL
    
    返回：
        list: 引用字典列表，每个字典包含以下键：
              - "start_index" (int): 引用片段在原始文本中的起始字符索引
                                    如果未指定则默认为0
              - "end_index" (int): 引用片段结束后的字符索引（不包含）
              - "segments" (list): 每个grounding块的Markdown格式链接列表
              - "segment_string" (str): 所有Markdown格式链接的连接字符串
              如果没有找到有效的候选或grounding支持，或缺少必要数据，
              则返回空列表
    """
    citations = []  # 初始化引用列表

    # 确保响应和必要的嵌套结构存在
    if not response or not response.candidates:
        return citations

    candidate = response.candidates[0]  # 获取第一个候选响应
    if (
        not hasattr(candidate, "grounding_metadata")
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, "grounding_supports")
    ):
        return citations

    # 遍历所有grounding支持信息
    for support in candidate.grounding_metadata.grounding_supports:
        citation = {}  # 初始化单个引用对象

        # 确保片段信息存在
        if not hasattr(support, "segment") or support.segment is None:
            continue  # 如果缺少片段信息则跳过此支持

        # 获取起始索引，如果为None则默认为0
        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        # 确保end_index存在以形成有效片段
        if support.segment.end_index is None:
            continue  # 如果缺少end_index则跳过，因为这是关键信息

        # 设置引用的起始和结束索引
        # (假设API提供的是包含性的end_index)
        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index

        citation["segments"] = []  # 初始化引用片段列表
        # 检查是否存在grounding块索引
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            # 遍历所有grounding块索引
            for ind in support.grounding_chunk_indices:
                try:
                    # 获取对应的grounding块
                    chunk = candidate.grounding_metadata.grounding_chunks[ind]
                    # 从URL映射中获取解析后的短URL
                    resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                    # 构建引用片段信息
                    citation["segments"].append(
                        {
                            "label": chunk.web.title.split(".")[:-1][0],  # 提取标题（去除扩展名）
                            "short_url": resolved_url,  # 短URL
                            "value": chunk.web.uri,  # 原始URI
                        }
                    )
                except (IndexError, AttributeError, NameError):
                    # 处理块、web、uri或resolved_map可能有问题的情况
                    # 为简单起见，我们只是跳过添加这个特定的片段链接
                    # 在生产系统中，您可能希望记录此错误
                    pass
        citations.append(citation)  # 将完整的引用添加到列表中
    return citations  # 返回所有提取的引用
