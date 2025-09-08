#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API端点测试脚本

测试双模型问答API端点的功能
"""

import asyncio
import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ 已加载.env文件")
except ImportError:
    print("⚠️  python-dotenv未安装，尝试手动加载.env文件")
    # 手动加载.env文件
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ 手动加载.env文件完成")
    else:
        print("❌ .env文件不存在")


async def test_api_endpoint():
    """测试API端点"""
    print("\n=== 测试双模型问答API端点 ===")
    
    try:
        import httpx
        
        # API请求数据
        request_data = {
            "message": "请简单介绍一下人工智能的发展历程。",
            "conversation_history": []
        }
        
        print(f"发送API请求: {request_data['message']}")
        
        # 发送请求到API端点
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://127.0.0.1:2024/api/dual-model-chat",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\n=== API响应结果 ===")
                print(f"成功状态: {result.get('success', False)}")
                print(f"处理阶段: {result.get('processing_stage', 'unknown')}")
                
                gemini_resp = result.get('gemini_response', '无回答')
                siliconflow_resp = result.get('siliconflow_response', '无回答')
                integrated_resp = result.get('integrated_response', '整合失败')
                
                print(f"\n【Gemini回答】: {gemini_resp[:100]}{'...' if len(gemini_resp) > 100 else ''}")
                print(f"\n【硅基流动回答】: {siliconflow_resp[:100]}{'...' if len(siliconflow_resp) > 100 else ''}")
                print(f"\n【整合回答】: {integrated_resp[:100]}{'...' if len(integrated_resp) > 100 else ''}")
                
                if result.get('success'):
                    print("\n✅ API端点测试成功！")
                    return True
                else:
                    print(f"\n❌ API返回失败: {result.get('error_message', '未知错误')}")
                    return False
            else:
                print(f"\n❌ API请求失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                return False
                
    except ImportError:
        print("\n⚠️  httpx未安装，跳过API端点测试")
        print("可以运行: pip install httpx 来安装")
        return None
    except Exception as e:
        print(f"\n❌ API测试过程中发生错误: {str(e)}")
        return False


async def test_health_endpoint():
    """测试健康检查端点"""
    print("\n=== 测试健康检查端点 ===")
    
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("http://127.0.0.1:2024/api/health")
            
            if response.status_code == 200:
                result = response.json()
                print(f"健康状态: {result.get('status', 'unknown')}")
                print(f"服务信息: {result.get('service', 'unknown')}")
                print("✅ 健康检查端点正常")
                return True
            else:
                print(f"❌ 健康检查失败，状态码: {response.status_code}")
                return False
                
    except ImportError:
        print("⚠️  httpx未安装，跳过健康检查测试")
        return None
    except Exception as e:
        print(f"❌ 健康检查测试失败: {str(e)}")
        return False


async def main():
    """主测试函数"""
    print("🚀 开始API端点测试")
    print("=" * 50)
    
    # 测试健康检查端点
    health_result = await test_health_endpoint()
    
    # 测试双模型问答API
    api_result = await test_api_endpoint()
    
    # 总结测试结果
    print("\n" + "=" * 50)
    print("📊 API测试结果总结")
    
    if health_result:
        print("健康检查: ✅ 通过")
    elif health_result is None:
        print("健康检查: ⚠️  跳过（缺少依赖）")
    else:
        print("健康检查: ❌ 失败")
    
    if api_result:
        print("双模型API: ✅ 通过")
    elif api_result is None:
        print("双模型API: ⚠️  跳过（缺少依赖）")
    else:
        print("双模型API: ❌ 失败")
    
    # 整体评估
    if health_result and api_result:
        print("\n🎉 所有API端点测试通过！")
    elif (health_result is None or health_result) and (api_result is None or api_result):
        print("\n⚠️  部分测试跳过，但可用测试均通过")
    else:
        print("\n❌ 部分API测试失败")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())