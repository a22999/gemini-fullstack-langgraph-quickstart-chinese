// =============================================================================
// 前端主应用组件 - App.tsx
// =============================================================================
// 本文件是前端应用的主入口组件，负责：
// 1. 与后端LangGraph代理建立WebSocket连接
// 2. 管理聊天消息和活动时间线状态
// 3. 处理用户输入和配置参数
// 4. 渲染欢迎界面或聊天界面
// 5. 实时显示代理执行过程中的各个步骤
// =============================================================================

import { useStream } from "@langchain/langgraph-sdk/react"; // LangGraph React SDK，用于流式连接
import type { Message } from "@langchain/langgraph-sdk"; // 消息类型定义
import { useState, useEffect, useRef, useCallback } from "react"; // React Hooks
import { ProcessedEvent } from "@/components/ActivityTimeline"; // 活动时间线事件类型
import { WelcomeScreen } from "@/components/WelcomeScreen"; // 欢迎界面组件
import { ChatMessagesView } from "@/components/ChatMessagesView"; // 聊天消息视图组件
import { Button } from "@/components/ui/button"; // UI按钮组件

// =============================================================================
// 主应用组件 - App
// =============================================================================
// 管理整个应用的状态和用户交互流程
// =============================================================================
export default function App() {
  // =========================================================================
  // 状态管理
  // =========================================================================
  
  // 当前活动时间线事件列表（实时更新）
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  
  // 历史活动记录（按消息ID存储）
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  
  // 滚动区域引用，用于自动滚动到底部
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  // 标记是否已发生最终化事件
  const hasFinalizeEventOccurredRef = useRef(false);
  
  // 错误状态
  const [error, setError] = useState<string | null>(null);
  // =========================================================================
  // LangGraph流式连接配置
  // =========================================================================
  // 建立与后端代理的WebSocket连接，实时接收执行状态更新
  const thread = useStream<{
    messages: Message[]; // 聊天消息列表
    initial_search_query_count: number; // 初始搜索查询数量
    max_research_loops: number; // 最大研究循环次数
    reasoning_model: string; // 推理模型名称
  }>({
    // API端点配置（开发环境使用2024端口，生产环境使用8123端口）
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "http://localhost:8123",
    assistantId: "agent", // 代理ID
    messagesKey: "messages", // 消息键名
    
    // =====================================================================
    // 实时事件更新处理器
    // =====================================================================
    // 监听后端代理执行过程中的各个步骤，转换为前端可显示的事件
    onUpdateEvent: (event: any) => {
      let processedEvent: ProcessedEvent | null = null;
      
      // 1. 查询生成流程事件
      if (event.generate_query) {
        processedEvent = {
          title: "生成搜索查询", // "Generating Search Queries"
          data: event.generate_query?.search_query?.join(", ") || "",
        };
      } 
      // 2. 网络研究流程事件
      else if (event.web_research) {
        const sources = event.web_research.sources_gathered || [];
        const numSources = sources.length;
        const uniqueLabels = [
          ...new Set(sources.map((s: any) => s.label).filter(Boolean)),
        ];
        const exampleLabels = uniqueLabels.slice(0, 3).join(", ");
        processedEvent = {
          title: "网络研究", // "Web Research"
          data: `收集了 ${numSources} 个来源。相关主题: ${
            exampleLabels || "无"
          }。`,
        };
      } 
      // 3. 反思评估流程事件
      else if (event.reflection) {
        processedEvent = {
          title: "反思评估", // "Reflection"
          data: "分析网络研究结果", // "Analysing Web Research Results"
        };
      } 
      // 4. 答案生成流程事件
      else if (event.finalize_answer) {
        processedEvent = {
          title: "生成最终答案", // "Finalizing Answer"
          data: "整理并呈现最终答案。", // "Composing and presenting the final answer."
        };
        hasFinalizeEventOccurredRef.current = true;
      }
      
      // 将处理后的事件添加到时间线
      if (processedEvent) {
        setProcessedEventsTimeline((prevEvents) => [
          ...prevEvents,
          processedEvent!,
        ]);
      }
    },
    
    // 错误处理器
    onError: (error: any) => {
      setError(error.message);
    },
  });

  // =========================================================================
  // 副作用处理
  // =========================================================================
  
  // 自动滚动到聊天底部
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [thread.messages]);

  // 保存历史活动记录
  // 当代理完成答案生成后，将当前时间线保存到历史记录中
  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current && // 已发生最终化事件
      !thread.isLoading && // 不在加载中
      thread.messages.length > 0 // 有消息存在
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        // 将当前时间线保存到历史活动记录中
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  // =========================================================================
  // 事件处理函数
  // =========================================================================
  
  // 处理用户提交查询
  const handleSubmit = useCallback(
    (submittedInputValue: string, effort: string, model: string) => {
      if (!submittedInputValue.trim()) return; // 空输入检查
      
      // 重置状态
      setProcessedEventsTimeline([]); // 清空当前时间线
      hasFinalizeEventOccurredRef.current = false; // 重置最终化标记

      // =====================================================================
      // 努力程度配置转换
      // =====================================================================
      // 将用户选择的努力程度转换为具体的搜索参数：
      // - low: 最多1个循环和1个查询（快速但简单）
      // - medium: 最多3个循环和3个查询（平衡）
      // - high: 最多10个循环和5个查询（深入但耗时）
      let initial_search_query_count = 0;
      let max_research_loops = 0;
      switch (effort) {
        case "low":
          initial_search_query_count = 1;
          max_research_loops = 1;
          break;
        case "medium":
          initial_search_query_count = 3;
          max_research_loops = 3;
          break;
        case "high":
          initial_search_query_count = 5;
          max_research_loops = 10;
          break;
      }

      // 构建新的消息列表
      const newMessages: Message[] = [
        ...(thread.messages || []), // 保留历史消息
        {
          type: "human", // 用户消息类型
          content: submittedInputValue, // 用户输入内容
          id: Date.now().toString(), // 生成唯一ID
        },
      ];
      
      // 提交到后端代理
      thread.submit({
        messages: newMessages,
        initial_search_query_count: initial_search_query_count,
        max_research_loops: max_research_loops,
        reasoning_model: model,
      });
    },
    [thread]
  );

  // 处理取消操作
  const handleCancel = useCallback(() => {
    thread.stop(); // 停止当前流式连接
    window.location.reload(); // 重新加载页面
  }, [thread]);

  // =========================================================================
  // 组件渲染
  // =========================================================================
  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="h-full w-full max-w-4xl mx-auto">
          {/* 根据应用状态渲染不同界面 */}
          {thread.messages.length === 0 ? (
            // 1. 欢迎界面 - 无消息时显示
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={thread.isLoading}
              onCancel={handleCancel}
            />
          ) : error ? (
            // 2. 错误界面 - 发生错误时显示
            <div className="flex flex-col items-center justify-center h-full">
              <div className="flex flex-col items-center justify-center gap-4">
                <h1 className="text-2xl text-red-400 font-bold">错误</h1>
                <p className="text-red-400">{JSON.stringify(error)}</p>

                <Button
                  variant="destructive"
                  onClick={() => window.location.reload()}
                >
                  重试
                </Button>
              </div>
            </div>
          ) : (
            // 3. 聊天界面 - 正常聊天时显示
            <ChatMessagesView
              messages={thread.messages} // 消息列表
              isLoading={thread.isLoading} // 加载状态
              scrollAreaRef={scrollAreaRef} // 滚动区域引用
              onSubmit={handleSubmit} // 提交处理器
              onCancel={handleCancel} // 取消处理器
              liveActivityEvents={processedEventsTimeline} // 实时活动事件
              historicalActivities={historicalActivities} // 历史活动记录
            />
          )}
      </main>
    </div>
  );
}
