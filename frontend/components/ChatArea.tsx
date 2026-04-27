import React, { useState, useRef, useEffect } from 'react';
import { streamChat, ChatMessage } from '../app/services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatAreaProps {
  model: string;
  apiKey: string;
  detail: string;
  proMode: boolean;
}

export default function ChatArea({ model, apiKey, detail, proMode }: ChatAreaProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const assistantMessage: ChatMessage = { role: 'assistant', content: '', sources: [], trace: {} };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      const stream = streamChat(input, model, apiKey, detail);
      
      for await (const chunk of stream) {
        if (chunk.type === 'sources') {
          assistantMessage.sources = chunk.data;
          setMessages((prev) => {
            const newMessages = [...prev];
            newMessages[newMessages.length - 1] = { ...assistantMessage };
            return newMessages;
          });
        } else if (chunk.type === 'content') {
          assistantMessage.content += chunk.data;
          setMessages((prev) => {
            const newMessages = [...prev];
            newMessages[newMessages.length - 1] = { ...assistantMessage };
            return newMessages;
          });
        } else if (chunk.type === 'trace' || chunk.type === 'trace_perf') {
          assistantMessage.trace = { ...assistantMessage.trace, ...chunk.data };
          setMessages((prev) => {
            const newMessages = [...prev];
            newMessages[newMessages.length - 1] = { ...assistantMessage };
            return newMessages;
          });
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      assistantMessage.content = '抱歉，发生了错误，请检查后端连接。';
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = { ...assistantMessage };
        return newMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mc-chat-area">
      <div className="mc-messages">
        <div className="mc-messages-inner">
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', marginTop: '100px', opacity: 0.7 }}>
              <h3 style={{ fontSize: '1.5rem', color: '#ffcc00' }}>✨ 欢迎来到 Minecraft 知识库</h3>
              <p>输入你的问题，例如：“红石中继器怎么合成？”</p>
            </div>
          )}
          {messages.map((msg, idx) => (
            <div key={idx} className={`mc-message ${msg.role}`}>
              <div className="message-header">
                {msg.role === 'user' ? '🧑‍🌾 Steve' : '🤖 Wiki Assistant'}
              </div>
              <div className="mc-markdown-content">
                {msg.role === 'assistant' ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content || (isLoading && idx === messages.length - 1 ? '正在挖掘数据...' : '')}
                  </ReactMarkdown>
                ) : (
                  <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                )}
              </div>

              {proMode && msg.trace && Object.keys(msg.trace).length > 0 && (
                <div className="mc-source-panel" style={{ backgroundColor: 'rgba(0, 0, 50, 0.4)', border: '1px dashed #555' }}>
                  <strong style={{ color: '#00ffcc' }}>🧪 专业模式 (Trace)</strong>
                  <ul style={{ listStyleType: 'none', padding: 0, margin: '8px 0', fontSize: '0.85rem' }}>
                    {msg.trace.search_terms && <li><strong>Query 改写:</strong> {msg.trace.search_terms.join(' | ')}</li>}
                    {msg.trace.retrieved_chunk_count !== undefined && <li><strong>召回分片数:</strong> {msg.trace.retrieved_chunk_count} 片</li>}
                    {msg.trace.retrieve_time_ms !== undefined && <li><strong>检索耗时:</strong> {msg.trace.retrieve_time_ms} ms</li>}
                    {msg.trace.ttft_ms !== undefined ? <li><strong>首字响应 (TTFT):</strong> {msg.trace.ttft_ms} ms</li> : null}
                    {msg.trace.total_time_ms !== undefined ? <li><strong>总计耗时:</strong> {msg.trace.total_time_ms} ms</li> : null}
                    {msg.trace.output_chars !== undefined ? <li><strong>生成字符:</strong> {msg.trace.output_chars} 字符</li> : null}
                  </ul>
                  
                  {msg.trace.system_prompt && (
                    <div style={{ marginTop: '10px' }}>
                      <details>
                        <summary style={{ cursor: 'pointer', color: '#ffcc00', fontSize: '0.85rem', fontWeight: 'bold' }}>[+] 查看 System Prompt</summary>
                        <pre style={{ fontSize: '0.75rem', whiteSpace: 'pre-wrap', backgroundColor: '#000', padding: '10px', marginTop: '8px', border: '1px solid #333' }}>
                          {msg.trace.system_prompt}
                        </pre>
                      </details>
                    </div>
                  )}

                  {msg.sources && msg.sources.length > 0 && (
                    <div style={{ marginTop: '10px' }}>
                      <details>
                        <summary style={{ cursor: 'pointer', color: '#ffcc00', fontSize: '0.85rem', fontWeight: 'bold' }}>[+] 预览检索上下文 ({msg.sources.length} 片段)</summary>
                        <div style={{ maxHeight: '400px', overflowY: 'auto', marginTop: '8px' }}>
                          {msg.sources.map((src, sIdx) => (
                            <div key={sIdx} style={{ fontSize: '0.75rem', marginBottom: '12px', padding: '10px', borderLeft: '3px solid #00ffcc', backgroundColor: '#0a0a0a' }}>
                              <div style={{ color: '#00ffcc', marginBottom: '5px', fontWeight: 'bold' }}>[{src.title} - {src.section}]</div>
                              <div style={{ whiteSpace: 'pre-wrap', color: '#ccc' }}>{src.content}</div>
                            </div>
                          ))}
                        </div>
                      </details>
                    </div>
                  )}
                </div>
              )}
              
              {msg.sources && msg.sources.length > 0 && (
                <div className="mc-source-panel">
                  <strong style={{ color: '#ffdd55' }}>📚 参考来源:</strong>
                  <ul>
                    {msg.sources.map((src, sIdx) => (
                      <li key={sIdx}>
                        <a href={src.source_url} target="_blank" rel="noreferrer">
                          {src.title} {src.section ? `- ${src.section}` : ''}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="mc-input-area">
        <form className="mc-input-form" onSubmit={handleSubmit}>
          <input
            className="mc-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="输入你想查询的 Minecraft 知识..."
            disabled={isLoading}
            autoFocus
          />
          <button type="submit" className="mc-button" disabled={isLoading}>
            {isLoading ? <span className="loading-dots">合成中</span> : '发送消息'}
          </button>
        </form>
      </div>
    </div>
  );
}
