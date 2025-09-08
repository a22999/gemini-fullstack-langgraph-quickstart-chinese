# =============================================================================
# LangGraph 智能研究代理 - 核心图定义文件
# =============================================================================
# 本文件定义了一个基于 LangGraph 的智能研究代理，实现以下核心流程：
# 1. 查询生成流程 → generate_query 函数
# 2. 并行分发流程 → continue_to_web_research 函数  
# 3. 网络研究流程 → web_research 函数
# 4. 反思评估流程 → reflection 函数
# 5. 路由决策流程 → evaluate_research 函数
# 6. 答案生成流程 → finalize_answer 函数
# =============================================================================

import os

# 导入结构化输出模型
from agent.tools_and_schemas import SearchQueryList, Reflection
# 环境变量加载
from dotenv import load_dotenv
# LangChain 核心组件
from langchain_core.messages import AIMessage
# LangGraph 图构建组件
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
# Google Gemini API 客户端
from google.genai import Client

# 状态管理模块
from agent.state import (
    OverallState,        # 全局状态
    QueryGenerationState, # 查询生成状态
    ReflectionState,     # 反思状态
    WebSearchState,      # 网络搜索状态
)
# 配置管理
from agent.configuration import Configuration
# 提示词模板
from agent.prompts import (
    get_current_date,
    query_writer_instructions,    # 查询生成提示词
    web_searcher_instructions,   # 网络搜索提示词
    reflection_instructions,     # 反思评估提示词
    answer_instructions,         # 答案生成提示词
)
# Google Gemini 聊天模型
from langchain_google_genai import ChatGoogleGenerativeAI
# 工具函数
from agent.utils import (
    get_citations,           # 获取引用信息
    get_research_topic,      # 提取研究主题
    insert_citation_markers, # 插入引用标记
    resolve_urls,           # 解析URL
)

# 加载环境变量
load_dotenv()

# 检查 API 密钥是否设置
if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# 初始化 Google Gemini API 客户端（用于 Google Search API）
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
# =============================================================================
# 1. 查询生成流程 → generate_query 函数
# =============================================================================
# 功能：基于用户问题生成多个搜索查询
# 输入：OverallState（包含用户消息）
# 输出：QueryGenerationState（包含搜索查询列表）
# 模型：使用配置中的查询生成模型（默认：gemini-2.0-flash）
# =============================================================================
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """【步骤1：查询生成流程】
    基于用户问题生成多个搜索查询，为后续的网络研究提供多角度的搜索方向。

    使用 Gemini 2.0 Flash 模型分析用户问题，生成多个优化的搜索查询。
    这些查询将用于并行的网络搜索，以获取全面的信息。

    Args:
        state: 全局状态，包含用户的原始问题和消息历史
        config: 运行时配置，包含模型选择、查询数量等参数

    Returns:
        QueryGenerationState: 包含生成的搜索查询列表的状态更新
    """
    # 从运行时配置中获取可配置参数
    configurable = Configuration.from_runnable_config(config)

    # 检查并设置初始搜索查询数量（如果未自定义则使用默认值）
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # 初始化 Gemini 2.0 Flash 模型
    # 温度设为1.0以增加查询的多样性和创造性
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,  # 查询生成模型（默认：gemini-2.0-flash）
        temperature=1.0,                          # 高温度值，增加查询多样性
        max_retries=2,                           # 最大重试次数
        api_key=os.getenv("GEMINI_API_KEY"),     # API密钥
    )
    # 配置结构化输出，确保返回符合SearchQueryList格式的结果
    structured_llm = llm.with_structured_output(SearchQueryList)

    # 格式化提示词模板
    current_date = get_current_date()  # 获取当前日期，用于时效性查询
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,                           # 当前日期
        research_topic=get_research_topic(state["messages"]), # 从消息中提取研究主题
        number_queries=state["initial_search_query_count"],  # 需要生成的查询数量
    )
    
    # 调用模型生成搜索查询
    # 模型会根据研究主题生成多个不同角度的搜索查询
    result = structured_llm.invoke(formatted_prompt)
    
    # 返回生成的搜索查询列表
    return {"search_query": result.query}


# =============================================================================
# 2. 并行分发流程 → continue_to_web_research 函数
# =============================================================================
# 功能：将生成的搜索查询分发到多个并行的网络研究节点
# 输入：QueryGenerationState（包含搜索查询列表）
# 输出：Send对象列表（用于并行执行网络搜索）
# 特点：实现并行处理，提高搜索效率
# =============================================================================
def continue_to_web_research(state: QueryGenerationState):
    """【步骤2：并行分发流程】
    将生成的多个搜索查询分发到并行的网络研究节点进行处理。
    
    这个函数是LangGraph的关键特性之一，它允许将单个状态分发到多个并行节点，
    每个节点处理一个搜索查询，从而实现高效的并行网络搜索。
    
    Args:
        state: 查询生成状态，包含从generate_query步骤生成的搜索查询列表
        
    Returns:
        list[Send]: Send对象列表，每个Send对象包含：
                   - 目标节点名称："web_research"
                   - 传递的数据：搜索查询和唯一ID
    """
    # 为每个搜索查询创建一个Send对象，实现并行分发
    # enumerate提供索引作为搜索ID，用于后续结果的追踪和管理
    return [
        Send(
            "web_research",  # 目标节点名称
            {
                "search_query": search_query,  # 具体的搜索查询
                "id": int(idx)                 # 搜索查询的唯一标识符
            }
        )
        for idx, search_query in enumerate(state["search_query"])  # 遍历所有搜索查询
    ]


# =============================================================================
# 3. 网络研究流程 → web_research 函数
# =============================================================================
# 功能：执行具体的网络搜索并收集结果
# 输入：WebSearchState（包含单个搜索查询和ID）
# 输出：包含搜索结果的状态更新
# 工具：使用Google Search API进行实际的网络搜索
# 特点：并行执行，每个查询独立处理
# =============================================================================
def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """【步骤3：网络研究流程】
    执行具体的网络搜索，使用Google Search API获取相关信息。
    
    这个函数是并行执行的核心，每个实例处理一个搜索查询，
    通过Google Search API获取实时的网络信息，并提取引用信息。

    Args:
        state: 网络搜索状态，包含：
               - search_query: 要搜索的查询字符串
               - id: 搜索的唯一标识符
        config: 运行时配置，包含模型选择等参数

    Returns:
        OverallState: 包含搜索结果的状态更新，包括：
                     - sources_gathered: 收集的信息源
                     - search_query: 搜索查询列表
                     - web_research_result: 搜索结果内容
    """
    # 从运行时配置中获取可配置参数
    configurable = Configuration.from_runnable_config(config)
    
    # 格式化网络搜索提示词模板
    # 包含当前日期和具体的搜索查询
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),        # 当前日期，用于时效性搜索
        research_topic=state["search_query"],   # 具体的搜索查询主题
    )

    # 使用Google GenAI客户端执行搜索
    # 注意：使用genai_client而不是langchain客户端，因为需要获取grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,  # 使用配置中的查询生成模型
        contents=formatted_prompt,                 # 格式化后的搜索提示词
        config={
            "tools": [{"google_search": {}}],     # 启用Google搜索工具
            "temperature": 0,                     # 温度设为0，确保搜索结果的一致性
        },
    )
    
    # 解析URL为短URL格式，节省token使用量和处理时间
    # grounding_metadata包含搜索结果的元数据信息
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, 
        state["id"]  # 使用搜索ID进行URL解析
    )
    
    # 获取引用信息并添加到生成的文本中
    # 这一步将搜索结果与具体的信息源关联起来
    citations = get_citations(response, resolved_urls)
    
    # 在生成的文本中插入引用标记
    # 确保每个信息都有对应的引用来源
    modified_text = insert_citation_markers(response.text, citations)
    
    # 提取所有引用片段，用于后续的信息源管理
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    # 返回更新的状态信息
    return {
        "sources_gathered": sources_gathered,        # 收集到的信息源列表
        "search_query": [state["search_query"]],     # 当前搜索查询（列表格式）
        "web_research_result": [modified_text],      # 带引用标记的搜索结果
    }


# =============================================================================
# 4. 反思评估流程 → reflection 函数
# =============================================================================
# 功能：评估当前研究结果的充分性，决定是否需要进一步研究
# 输入：OverallState（包含所有网络研究结果）
# 输出：ReflectionState（包含研究充分性判断和后续查询）
# 模型：使用配置中的反思模型（默认：gemini-2.5-flash）
# 特点：智能判断研究完整性，生成补充查询
# =============================================================================
def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """【步骤4：反思评估流程】
    分析当前收集的研究结果，评估信息的充分性和完整性。
    
    这个函数是智能研究系统的关键组件，它能够：
    1. 评估当前研究结果是否足够回答用户问题
    2. 识别知识空白和信息缺口
    3. 生成针对性的后续查询建议
    4. 决定是否需要进行额外的研究循环
    
    Args:
        state: 全局状态，包含：
               - web_research_result: 所有网络搜索结果
               - messages: 用户的原始问题
               - research_loop_count: 当前研究循环次数
        config: 运行时配置，包含反思模型选择等参数
        
    Returns:
        ReflectionState: 包含反思结果的状态更新：
                        - is_sufficient: 研究是否充分（布尔值）
                        - knowledge_gap: 识别出的知识空白
                        - follow_up_queries: 建议的后续查询列表
                        - research_loop_count: 更新的研究循环计数
    """
    # 从运行时配置中获取可配置参数
    configurable = Configuration.from_runnable_config(config)
    
    # 递增研究循环计数器，用于控制最大研究轮次
    # 防止无限循环，确保系统在合理时间内完成研究
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    
    # 获取推理模型配置
    # 优先使用状态中指定的模型，否则使用配置中的默认反思模型
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # 格式化反思评估提示词模板
    # 包含当前日期、研究主题和所有收集到的研究结果
    current_date = get_current_date()  # 获取当前日期，用于时效性评估
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,                                      # 当前日期
        research_topic=get_research_topic(state["messages"]),           # 从消息中提取研究主题
        summaries="\n\n---\n\n".join(state["web_research_result"]),    # 合并所有搜索结果
    )
    
    # 初始化反思评估模型
    # 使用较高温度(1.0)以获得更有创造性的知识空白识别和后续查询生成
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,                   # 反思模型（默认：gemini-2.5-flash）
        temperature=1.0,                        # 高温度值，增加创造性思考
        max_retries=2,                          # 最大重试次数
        api_key=os.getenv("GEMINI_API_KEY"),    # API密钥
    )
    
    # 执行结构化反思评估
    # 使用Reflection模式确保返回标准化的评估结果
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    # 返回结构化的反思评估结果
    return {
        "is_sufficient": result.is_sufficient,                    # 当前研究是否充分的判断
        "knowledge_gap": result.knowledge_gap,                    # 识别出的主要知识空白
        "follow_up_queries": result.follow_up_queries,            # 建议的后续查询列表
        "research_loop_count": state["research_loop_count"],      # 更新的研究循环计数
        "number_of_ran_queries": len(state["search_query"]),      # 已执行的查询总数
    }


# =============================================================================
# 5. 路由决策流程 → evaluate_research 函数
# =============================================================================
# 功能：基于反思评估结果决定下一步行动路径
# 输入：ReflectionState（包含反思评估结果和循环计数）
# 输出：字符串或Send对象列表（下一个节点或并行任务）
# 决策逻辑：研究充分性 + 最大循环次数限制
# 路径选择："finalize_answer" 或 继续网络研究
# =============================================================================
def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """【步骤5：路由决策流程】
    基于反思评估的结果，智能决定研究流程的下一步行动。
    
    这个函数实现了智能路由逻辑，它会综合考虑以下因素：
    1. 研究充分性：当前信息是否足够回答用户问题
    2. 循环次数限制：防止无限循环，确保系统效率
    3. 后续查询生成：基于知识空白生成针对性的补充查询
    
    决策路径：
    - 如果研究充分 OR 达到最大循环次数 → 进入答案生成阶段
    - 如果研究不充分 AND 未达到最大循环次数 → 生成后续查询继续研究
    
    Args:
        state: 反思状态，包含：
               - is_sufficient: 研究充分性判断（布尔值）
               - research_loop_count: 当前研究循环次数
               - follow_up_queries: 建议的后续查询列表
               - number_of_ran_queries: 已执行的查询总数
        config: 运行时配置，包含：
                - max_research_loops: 最大研究循环次数限制
               
    Returns:
        Union[str, List[Send]]: 路由决策结果：
                               - "finalize_answer": 进入最终答案生成阶段
                               - List[Send]: 并行执行的后续网络研究任务列表
    """
    # 从运行时配置中获取可配置参数
    configurable = Configuration.from_runnable_config(config)
    
    # 确定最大研究循环次数限制
    # 优先使用状态中指定的值，否则使用配置中的默认值
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    
    # 智能路由决策逻辑
    # 条件1：反思评估认为当前研究结果已经充分
    # 条件2：达到最大研究循环次数限制，防止无限循环和资源浪费
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        # 研究完成，进入最终答案生成阶段
        return "finalize_answer"
    else:
        # 研究不充分，需要继续进行网络研究
        # 基于反思评估中识别的知识空白，生成针对性的后续查询
        # 使用Send对象实现并行执行多个后续查询
        return [
            Send(
                "web_research",  # 目标节点：网络研究
                {
                    "search_query": follow_up_query,  # 后续查询内容
                    "id": state["number_of_ran_queries"] + int(idx),  # 分配唯一ID
                },
            )
            # 遍历所有建议的后续查询，为每个查询创建一个并行任务
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


# =============================================================================
# 6. 答案生成流程 → finalize_answer 函数
# =============================================================================
# 功能：基于所有收集的研究结果生成最终答案
# 输入：OverallState（包含所有网络研究结果和信息源）
# 输出：包含最终答案和引用源的状态更新
# 模型：使用配置中的答案生成模型（默认：gemini-2.5-flash）
# 特点：整合所有信息，生成结构化的最终回答
# =============================================================================
def finalize_answer(state: OverallState, config: RunnableConfig):
    """【步骤6：答案生成流程】
    整合所有收集的研究结果，生成全面、准确的最终答案。
    
    这个函数是整个研究流程的最终阶段，它会：
    1. 整合所有网络搜索结果和信息源
    2. 使用高质量的答案生成模型进行综合分析
    3. 生成结构化、有引用的最终答案
    4. 处理URL替换和信息源去重，确保可追溯性
    
    Args:
        state: 全局状态，包含：
               - web_research_result: 所有网络搜索结果
               - sources_gathered: 收集的信息源列表
               - messages: 用户的原始问题
               - reasoning_model: 可选的推理模型覆盖
        config: 运行时配置，包含答案生成模型选择等参数
        
    Returns:
        dict: 包含最终答案的状态更新：
              - messages: 包含AI回答的消息列表
              - sources_gathered: 去重后的信息源列表
    """
    # 从运行时配置中获取可配置参数
    configurable = Configuration.from_runnable_config(config)
    
    # 确定答案生成模型
    # 优先使用状态中指定的推理模型，否则使用配置中的默认答案模型
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # 格式化答案生成提示词模板
    # 包含当前日期、研究主题和所有收集到的研究结果
    current_date = get_current_date()  # 获取当前日期，用于时效性标注
    formatted_prompt = answer_instructions.format(
        current_date=current_date,                                      # 当前日期
        research_topic=get_research_topic(state["messages"]),           # 从消息中提取研究主题
        summaries="\n---\n\n".join(state["web_research_result"]),      # 合并所有搜索结果
    )

    # 初始化答案生成模型（默认使用 Gemini 2.5 Flash）
    # 温度设为0以确保答案的准确性和一致性
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,                   # 答案生成模型
        temperature=0,                          # 温度设为0，确保答案的准确性
        max_retries=2,                          # 最大重试次数
        api_key=os.getenv("GEMINI_API_KEY"),    # API密钥
    )
    
    # 调用模型生成最终答案
    # 基于整合的研究结果和格式化的提示词生成综合性回答
    result = llm.invoke(formatted_prompt)

    # 处理URL替换和信息源收集
    # 将答案中的短URL替换为原始URL，并收集实际使用的信息源
    unique_sources = []
    for source in state["sources_gathered"]:
        # 检查答案内容中是否包含该信息源的短URL
        if source["short_url"] in result.content:
            # 将短URL替换为原始完整URL，提高可读性和可访问性
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            # 将实际使用的信息源添加到最终的源列表中
            unique_sources.append(source)

    # 返回最终的答案和使用的信息源
    return {
        "messages": [AIMessage(content=result.content)],  # AI生成的最终答案
        "sources_gathered": unique_sources,              # 实际使用的信息源列表
    }


# =============================================================================
# LangGraph 图构建与编译
# =============================================================================
# 本节定义了整个智能研究代理的执行流程图，包括：
# 1. 节点定义：各个处理步骤的具体实现
# 2. 边连接：节点之间的数据流转和执行顺序
# 3. 条件路由：基于状态的智能决策分支
# 4. 图编译：将逻辑图转换为可执行的工作流
# =============================================================================

# 初始化状态图构建器
# 使用 OverallState 作为全局状态管理器，Configuration 作为配置模式
builder = StateGraph(OverallState, config_schema=Configuration)

# =============================================================================
# 节点注册：将各个功能函数注册为图中的处理节点
# =============================================================================

# 注册查询生成节点：负责将用户问题转换为多个搜索查询
builder.add_node("generate_query", generate_query)

# 注册网络研究节点：负责执行并行的网络搜索和信息收集
builder.add_node("web_research", web_research)

# 注册反思评估节点：负责分析研究结果的充分性和生成后续查询
builder.add_node("reflection", reflection)

# 注册答案生成节点：负责整合所有信息并生成最终答案
builder.add_node("finalize_answer", finalize_answer)

# =============================================================================
# 边连接定义：建立节点间的执行流程和数据传递路径
# =============================================================================

# 设置起始边：从图的开始点直接进入查询生成阶段
# 这意味着 generate_query 是整个流程的第一个被调用的节点
builder.add_edge(START, "generate_query")

# 添加条件边：查询生成 → 并行网络研究
# 使用 continue_to_web_research 函数将查询分发到多个并行的网络研究节点
builder.add_conditional_edges(
    "generate_query",           # 源节点：查询生成
    continue_to_web_research,    # 路由函数：并行分发逻辑
    ["web_research"]            # 目标节点列表：网络研究（支持并行执行）
)

# 添加直接边：网络研究 → 反思评估
# 所有并行的网络研究完成后，自动进入反思评估阶段
builder.add_edge("web_research", "reflection")

# 添加条件边：反思评估 → 智能路由决策
# 基于研究充分性和循环次数，决定是继续研究还是生成最终答案
builder.add_conditional_edges(
    "reflection",               # 源节点：反思评估
    evaluate_research,          # 路由函数：智能决策逻辑
    ["web_research", "finalize_answer"]  # 可能的目标节点：继续研究或生成答案
)

# 添加结束边：答案生成 → 图结束
# 最终答案生成完成后，整个研究流程结束
builder.add_edge("finalize_answer", END)

# =============================================================================
# 图编译：将逻辑图转换为可执行的 LangGraph 工作流
# =============================================================================
# 编译过程会：
# 1. 验证图的完整性和一致性
# 2. 优化执行路径和并行处理能力
# 3. 生成可执行的状态机
# 4. 支持检查点、重试和错误处理机制
graph = builder.compile(name="pro-search-agent")

# =============================================================================
# 图执行流程总结
# =============================================================================
# 完整的执行路径：
# START → generate_query → [并行] web_research → reflection → 决策分支
#                                                              ↓
#                                                    ┌─ 继续研究（循环）
#                                                    └─ finalize_answer → END
# 
# 关键特性：
# 1. 并行处理：多个搜索查询同时执行，提高效率
# 2. 智能路由：基于研究质量动态调整流程
# 3. 循环控制：防止无限循环，确保系统稳定性
# 4. 状态管理：全程跟踪数据流转和处理状态
# 5. 错误恢复：支持检查点和重试机制
# =============================================================================
