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
from fastapi import FastAPI, Response  # FastAPI框架和响应类
from fastapi.staticfiles import StaticFiles  # 静态文件服务

# 定义FastAPI应用实例
app = FastAPI()


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
