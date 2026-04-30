"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ArrowUp, BookText, Share2, Save, Search, Database, Loader2, CheckCircle2 } from "lucide-react";
import {
  ChatMessage,
  streamChat,
  getKnowledgeBaseStatus,
  getEmbeddingModels,
  switchEmbeddingModel,
  KnowledgeBaseStatus,
  EmbeddingModelInfo,
} from "./services/api";
import LeftSidebar from "../components/LeftSidebar";
import RightSidebar from "../components/RightSidebar";
import ProfessionalTrace from "../components/ProfessionalTrace";

import { Button } from "../components/ui/button";

function loadMessages(): ChatMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = sessionStorage.getItem("rag-chat-messages");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>(loadMessages);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [kbStatus, setKbStatus] = useState<KnowledgeBaseStatus | null>(null);

  const [modelChoice, setModelChoice] = useState("api");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [modelName, setModelName] = useState("");
  const [answerDetail, setAnswerDetail] = useState("标准");
  const [professionalMode, setProfessionalMode] = useState(false);
  const [embeddingModelId, setEmbeddingModelId] = useState("");
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModelInfo[]>([]);

  const chatContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isAutoScrolling = useRef(true);

  useEffect(() => {
    sessionStorage.setItem("rag-chat-messages", JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    const checkStatus = async () => {
      try {
        const status = await getKnowledgeBaseStatus();
        setKbStatus(status);
        if (status.available_models) {
          setEmbeddingModels(status.available_models);
          const active = status.available_models.find((m) => m.is_active);
          if (active) setEmbeddingModelId(active.id);
        }
        if (status.status !== "ready") {
          timer = setTimeout(checkStatus, 3000);
        }
      } catch (e) {
        console.error("获取状态失败", e);
        timer = setTimeout(checkStatus, 5000);
      }
    };
    checkStatus();
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [inputValue]);

  const handleSwitchEmbeddingModel = async (modelId: string) => {
    try {
      const status = await switchEmbeddingModel(modelId);
      setKbStatus(status);
      setEmbeddingModelId(modelId);
      if (status.available_models) setEmbeddingModels(status.available_models);
    } catch (e) {
      console.error("切换嵌入模型失败", e);
    }
  };

  const handleScroll = useCallback(() => {
    const container = chatContainerRef.current;
    if (!container) return;
    const isAtBottom =
      container.scrollHeight - container.scrollTop <= container.clientHeight + 100;
    isAutoScrolling.current = isAtBottom;
  }, []);

  useEffect(() => {
    if (isAutoScrolling.current && chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: ChatMessage = { role: "user", content: inputValue };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);
    isAutoScrolling.current = true;

    const assistantMessage: ChatMessage = {
      role: "assistant",
      content: "",
      trace: { user_input: userMessage.content, model_choice: modelChoice, answer_detail: answerDetail },
    };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      let currentContent = "";
      for await (const chunk of streamChat(
        userMessage.content,
        modelChoice,
        apiKey || undefined,
        answerDetail,
        baseUrl,
        modelName
      )) {
        if (chunk.type === "content") {
          currentContent += chunk.data;
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            return [...prev.slice(0, -1), { ...last, content: currentContent }];
          });
        } else if (chunk.type === "sources") {
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            return [...prev.slice(0, -1), { ...last, sources: chunk.data }];
          });
        } else if (chunk.type === "trace" || chunk.type === "trace_config") {
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            return [...prev.slice(0, -1), { ...last, trace: { ...last.trace, ...chunk.data } }];
          });
        } else if (chunk.type === "trace_perf") {
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            return [...prev.slice(0, -1), { ...last, trace: { ...last.trace, ...chunk.data } }];
          });
        }
      }
    } catch (error) {
      console.error("对话错误:", error);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", content: "抱歉，连接后端时出现错误，请检查网络或服务状态。" },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#f4f4f5]">
      <LeftSidebar
        kbStatus={kbStatus}
        onRefresh={async () => {
          try {
            const status = await getKnowledgeBaseStatus();
            setKbStatus(status);
            if (status.available_models) setEmbeddingModels(status.available_models);
          } catch (e) {
            console.error("刷新状态失败", e);
          }
        }}
        embeddingModelId={embeddingModelId}
        embeddingModels={embeddingModels}
        onSwitchEmbeddingModel={handleSwitchEmbeddingModel}
      />

      <main className="relative flex flex-1 flex-col bg-card">
        {/* Header */}
        <header className="z-10 flex h-16 shrink-0 items-center justify-between border-b border-border px-6">
          <span className="text-sm font-medium text-foreground">推理控制台</span>
          <div className="flex gap-3">
            <Button variant="outline" size="sm">
              <Save size={16} />
              保存
            </Button>
            <Button variant="outline" size="sm">
              <Share2 size={16} />
              分享
            </Button>
          </div>
        </header>

        {/* Chat Area */}
        <div className="relative flex flex-1 flex-col overflow-hidden">
          <div
            className="flex flex-1 flex-col overflow-y-auto scroll-smooth"
            ref={chatContainerRef}
            onScroll={handleScroll}
          >
            {messages.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center p-20 text-center opacity-60">
                <BookText size={56} strokeWidth={1} />
                <h2 className="mt-4 text-2xl font-semibold">Minecraft RAG 知识引擎</h2>
                <p className="mt-3 max-w-md text-base leading-relaxed text-muted-foreground">
                  输入您的 Minecraft 疑问，系统将检索 Wiki 数据库并生成专业解答。
                </p>
              </div>
            ) : (
              <div className="mx-auto flex w-full max-w-3xl flex-col gap-16 px-8 py-16">
                {messages.map((msg, i) => (
                  <div key={i} className={`flex flex-col ${msg.role}`}>
                    {/* User message */}
                    {msg.role === "user" && (
                      <div className="mb-6 border-b border-border pb-8">
                        <div className="mb-4 text-[11px] font-bold tracking-widest text-muted-foreground">
                          USER
                        </div>
                        <div className="text-2xl font-semibold leading-tight text-foreground">
                          {msg.content}
                        </div>
                      </div>
                    )}

                    {/* Assistant message */}
                    {msg.role === "assistant" && (
                      <div className="flex gap-10">
                        {/* Main content column */}
                        <div className="relative flex-1 border-l-2 border-border pl-8">
                          {/* Thought process */}
                          <div className="mb-8 flex flex-col gap-3">
                            {/* Query rewrite step */}
                            <div
                              className={`flex items-center gap-3 text-sm transition-all duration-500 ${
                                msg.trace?.search_terms
                                  ? "opacity-80 text-muted-foreground"
                                  : "font-medium text-foreground opacity-100"
                              }`}
                            >
                              <span className="flex size-4 shrink-0 items-center justify-center">
                                {msg.trace?.search_terms ? (
                                  <CheckCircle2 size={14} className="text-muted-foreground" />
                                ) : (
                                  <Loader2 size={14} className="custom-spin" />
                                )}
                              </span>
                              <span>查询改写</span>
                              {msg.trace?.search_terms && (
                                <span className="ml-1 rounded bg-muted px-2 py-0.5 font-mono text-xs text-muted-foreground">
                                  {msg.trace.search_terms.join(", ")}
                                </span>
                              )}
                            </div>

                            {/* Retrieval step */}
                            <div
                              className={`flex items-center gap-3 text-sm transition-all duration-500 ${
                                msg.sources
                                  ? "opacity-80 text-muted-foreground"
                                  : msg.trace?.search_terms
                                    ? "font-medium text-foreground opacity-100"
                                    : "text-muted-foreground opacity-20"
                              }`}
                            >
                              <span className="flex size-4 shrink-0 items-center justify-center">
                                {msg.sources ? (
                                  <CheckCircle2 size={14} className="text-muted-foreground" />
                                ) : msg.trace?.search_terms ? (
                                  <Loader2 size={14} className="custom-spin" />
                                ) : (
                                  <Database size={14} />
                                )}
                              </span>
                              <span>知识检索</span>
                              {msg.trace?.retrieved_chunk_count !== undefined && (
                                <span className="ml-1 rounded bg-muted px-2 py-0.5 font-mono text-xs text-muted-foreground">
                                  获取核心切片 {msg.trace.retrieved_chunk_count} 条
                                </span>
                              )}
                            </div>

                            {/* Answer generation step */}
                            <div
                              className={`flex items-center gap-3 text-sm transition-all duration-500 ${
                                msg.content
                                  ? "opacity-80 text-muted-foreground"
                                  : msg.sources
                                    ? "font-medium text-foreground opacity-100"
                                    : "text-muted-foreground opacity-20"
                              }`}
                            >
                              <span className="flex size-4 shrink-0 items-center justify-center">
                                {msg.content ? (
                                  <CheckCircle2 size={14} className="text-muted-foreground" />
                                ) : msg.sources ? (
                                  <Loader2 size={14} className="custom-spin" />
                                ) : (
                                  <Search size={14} />
                                )}
                              </span>
                              <span>回答生成</span>
                            </div>
                          </div>

                          {/* Answer content */}
                          <div className="mb-8 text-[17px] leading-relaxed text-foreground [&_p]:mb-6">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {msg.content}
                            </ReactMarkdown>
                          </div>

                          {/* Sources */}
                          {msg.sources && msg.sources.length > 0 && (
                            <div className="border-t border-border pt-6">
                              <div className="mb-4 text-[11px] font-bold tracking-wider text-muted-foreground">
                                参考文献
                              </div>
                              <div className="flex flex-wrap gap-2.5">
                                {msg.sources.map((src, j) => (
                                  <a
                                    key={j}
                                    href={src.source_url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="rounded-md border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
                                  >
                                    {src.title} · {src.section}
                                    {src.content_length !== undefined && (
                                      <span className="ml-1 text-[11px] text-muted-foreground/60">
                                        ({src.content_length}字)
                                      </span>
                                    )}
                                  </a>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Professional trace */}
                          {professionalMode && msg.trace && msg.trace.search_terms && (
                            <ProfessionalTrace trace={msg.trace} sources={msg.sources} />
                          )}
                        </div>

                        {/* Status aside */}
                        <div className="flex w-10 flex-col items-end pt-2">
                          {isLoading && i === messages.length - 1 ? (
                            <Loader2 className="custom-spin text-foreground" size={20} />
                          ) : (
                            <div className="flex size-6 items-center justify-center">
                              <div className="size-1 rounded-full bg-muted-foreground/30" />
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
                <div className="h-10" />
              </div>
            )}
          </div>

          {/* Fade gradient at bottom */}
          <div className="pointer-events-none absolute bottom-0 left-0 right-0  z-10h-12 bg-linear-to-b from-transparent to-card" />
        </div>

        {/* Input area */}
        <div className="z-10 shrink-0 px-8 pb-8 pt-2">
          <form onSubmit={handleSubmit} className="mx-auto max-w-176">
            <div className="flex items-end gap-2.5 rounded-2xl border border-border bg-card px-4 py-3.5 shadow-sm transition-all duration-200 focus-within:border-foreground focus-within:shadow-md">
              <textarea
                ref={textareaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder="在此输入您的 Minecraft 疑问..."
                rows={1}
                disabled={isLoading}
                className="flex-1 resize-none self-center border-none bg-transparent py-0.5 text-[17px] leading-relaxed text-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0"
                style={{ maxHeight: "12rem", minHeight: "1.5rem", fieldSizing: "content" }}
              />
              <button
                type="submit"
                title="发送"
                disabled={isLoading || !inputValue.trim()}
                className={`flex size-10 shrink-0 items-center justify-center rounded-xl transition-all ${
                  inputValue.trim() && !isLoading
                    ? "bg-foreground text-white"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                {isLoading ? <Loader2 className="custom-spin" size={18} /> : <ArrowUp size={18} strokeWidth={2.5} />}
              </button>
            </div>
            <div className="mt-4 flex justify-center text-[11px] font-medium tracking-wider text-muted-foreground">
              由 RAG-MCWIKI 引擎驱动
            </div>
          </form>
        </div>
      </main>

      <RightSidebar
        modelChoice={modelChoice}
        setModelChoice={setModelChoice}
        baseUrl={baseUrl}
        setBaseUrl={setBaseUrl}
        apiKey={apiKey}
        setApiKey={setApiKey}
        modelName={modelName}
        setModelName={setModelName}
        answerDetail={answerDetail}
        setAnswerDetail={setAnswerDetail}
        professionalMode={professionalMode}
        setProfessionalMode={setProfessionalMode}

      />
    </div>
  );
}
