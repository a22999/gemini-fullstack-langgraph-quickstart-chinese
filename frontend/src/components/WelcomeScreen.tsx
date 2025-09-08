// =============================================================================
// 欢迎界面组件 - WelcomeScreen.tsx
// =============================================================================
// 应用的初始欢迎界面，在用户尚未开始对话时显示
// 提供友好的欢迎信息和输入表单
// =============================================================================

import { InputForm } from "./InputForm"; // 输入表单组件

// =============================================================================
// 组件属性接口定义
// =============================================================================
interface WelcomeScreenProps {
  handleSubmit: (
    submittedInputValue: string, // 用户输入的查询内容
    effort: string, // 努力程度（low/medium/high）
    model: string // 选择的AI模型
  ) => void;
  onCancel: () => void; // 取消操作回调
  isLoading: boolean; // 是否正在加载
}

// =============================================================================
// 欢迎界面组件
// =============================================================================
// 展示欢迎信息和输入表单，引导用户开始对话
// =============================================================================
export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  handleSubmit,
  onCancel,
  isLoading,
}) => (
  <div className="h-full flex flex-col items-center justify-center text-center px-4 flex-1 w-full max-w-3xl mx-auto gap-4">
    {/* 欢迎标题区域 */}
    <div>
      <h1 className="text-5xl md:text-6xl font-semibold text-neutral-100 mb-3">
        欢迎使用 {/* Welcome. */}
      </h1>
      <p className="text-xl md:text-2xl text-neutral-400">
        今天我可以为您做些什么？ {/* How can I help you today? */}
      </p>
    </div>
    
    {/* 输入表单区域 */}
    <div className="w-full mt-4">
      <InputForm
        onSubmit={handleSubmit} // 提交处理器
        isLoading={isLoading} // 加载状态
        onCancel={onCancel} // 取消处理器
        hasHistory={false} // 无历史记录（首次使用）
      />
    </div>
    
    {/* 技术支持信息 */}
    <p className="text-xs text-neutral-500">
      由 Google Gemini 和 LangChain LangGraph 提供支持。 {/* Powered by Google Gemini and LangChain LangGraph. */}
    </p>
  </div>
);
