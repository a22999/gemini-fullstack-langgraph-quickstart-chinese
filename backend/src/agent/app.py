# -*- coding: utf-8 -*-
# mypy: disable - error - code = "no-untyped-def,misc"
"""
FastAPI应用主入口 (FastAPI Application Main Entry)

本文件定义了LangGraph研究代理的FastAPI应用程序，主要功能包括：
1. 创建FastAPI应用实例
2. 配置前端静态文件服务路由
3. 处理前端构建状态检查和错误处理
4. 将前端挂载到/app路径，避免与LangGraph API路由冲突

该应用作为全栈应用的后端服务，同时提供API接口和前端静态文件服务。
"""

import pathlib  # 路径处理工具
from fastapi import FastAPI, Response, HTTPException  # FastAPI框架和响应类
from fastapi.staticfiles import StaticFiles  # 静态文件服务
from pydantic import BaseModel  # 数据验证模型
from typing import List, Dict, Any  # 类型注解
from langchain_core.messages import HumanMessage, AIMessage  # LangChain消息类型

# 导入双模型聊天图
from src.chat.graph import dual_model_chat_graph
from src.chat.state import DualModelState

# 定义FastAPI应用实例
app = FastAPI()


# =============================================================================
# API数据模型定义
# =============================================================================

class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str  # "user" 或 "assistant"
    content: str  # 消息内容


class DualModelChatRequest(BaseModel):
    """双模型聊天请求模型"""
    message: str  # 用户输入的消息
    conversation_history: List[ChatMessage] = []  # 对话历史


class DualModelChatResponse(BaseModel):
    """双模型聊天响应模型"""
    gemini_response: str  # Gemini模型的回答
    siliconflow_response: str  # 硅基流动模型的回答
    integrated_response: str  # 整合后的最终回答
    processing_stage: str  # 处理阶段
    success: bool  # 是否成功
    error_message: str | None = None  # 错误信息（如果有）


# =============================================================================
# API端点定义
# =============================================================================

@app.post("/api/dual-model-chat", response_model=DualModelChatResponse)
async def dual_model_chat(request: DualModelChatRequest):
    """【双模型问答API端点】
    
    接收用户消息，使用Gemini和硅基流动两个模型并行生成回答，
    然后整合两个回答生成最终的综合回答。
    
    Args:
        request: 包含用户消息和对话历史的请求
        
    Returns:
        DualModelChatResponse: 包含两个模型回答和整合回答的响应
    """
    try:
        # 构建消息历史
        messages = []
        
        # 添加历史对话
        for msg in request.conversation_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        # 添加当前用户消息
        messages.append(HumanMessage(content=request.message))
        
        # 创建初始状态
        initial_state = DualModelState(
            messages=messages,
            processing_stage="initial",
            gemini_response=None,
            siliconflow_response=None,
            integrated_response=None
        )
        
        # 执行双模型问答图
        result = await dual_model_chat_graph.ainvoke(initial_state)
        
        # 构建响应
        return DualModelChatResponse(
            gemini_response=result.get("gemini_response", "无回答"),
            siliconflow_response=result.get("siliconflow_response", "无回答"),
            integrated_response=result.get("integrated_response", "整合失败"),
            processing_stage=result.get("processing_stage", "unknown"),
            success=result.get("processing_stage") == "completed",
            error_message=None
        )
        
    except Exception as e:
        # 错误处理
        return DualModelChatResponse(
            gemini_response="模型调用失败",
            siliconflow_response="模型调用失败",
            integrated_response=f"双模型问答失败: {str(e)}",
            processing_stage="error",
            success=False,
            error_message=str(e)
        )


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "dual-model-chat-api"}


def create_frontend_router(build_dir="../frontend/dist"):
    """
    创建前端路由器以服务React前端 (Create Frontend Router to Serve React Frontend)
    
    功能说明：
    - 检查前端构建目录是否存在且完整
    - 如果构建目录不存在，返回一个显示错误信息的虚拟路由
    - 如果构建目录存在，返回静态文件服务器来提供前端文件
    - 支持HTML5路由模式（html=True）
    
    参数：
        build_dir (str): 相对于此文件的React构建目录路径，默认为"../frontend/dist"
    
    返回：
        StaticFiles或Route: 用于服务前端的Starlette应用程序
    """
    # 构建前端构建目录的绝对路径
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    # 检查构建目录是否存在且包含index.html文件
    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # 如果构建未准备好，返回一个虚拟路由
        from starlette.routing import Route

        async def dummy_frontend(request):
            """虚拟前端处理器，返回构建错误信息"""
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,  # 服务不可用状态码
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    # 返回静态文件服务器，支持HTML5路由
    return StaticFiles(directory=build_path, html=True)


# 将前端挂载到/app路径下，避免与LangGraph API路由冲突
# 这样设计使得：
# - LangGraph API可以使用根路径和其他路径
# - 前端应用通过/app路径访问
# - 两者可以在同一个服务器上和谐共存
app.mount(
    "/app",  # 前端应用的挂载路径
    create_frontend_router(),  # 前端路由器实例
    name="frontend",  # 路由器名称，用于内部引用
)
