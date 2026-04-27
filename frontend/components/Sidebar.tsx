import React, { useState, useEffect } from 'react';
import {
  buildKnowledgeBase,
  cleanKnowledgeBase,
  getKnowledgeBaseStatus,
  KnowledgeBaseStatus,
  syncKnowledgeBase,
} from '../app/services/api';

interface SidebarProps {
  model: string;
  setModel: (val: string) => void;
  apiKey: string;
  setApiKey: (val: string) => void;
  detail: string;
  setDetail: (val: string) => void;
  proMode: boolean;
  setProMode: (val: boolean) => void;
}

export default function Sidebar({
  model, setModel,
  apiKey, setApiKey,
  detail, setDetail,
  proMode, setProMode
}: SidebarProps) {
  const [embeddingModel, setEmbeddingModel] = useState('shibing624/text2vec-base-chinese');
  const [isBuilding, setIsBuilding] = useState(false);
  const [isCleaning, setIsCleaning] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [kbStatus, setKbStatus] = useState<KnowledgeBaseStatus | null>(null);

  const isBusy = isBuilding || isCleaning || isSyncing;

  const applyStatusData = (data: KnowledgeBaseStatus) => {
    setKbStatus(data);
    if (data.embedding_model) {
      setEmbeddingModel(data.embedding_model);
    }
  };

  const fetchKBStatusData = async () => {
    try {
      return await getKnowledgeBaseStatus();
    } catch (error) {
      console.error('Failed to fetch KB status:', error);
    }
    return null;
  };

  useEffect(() => {
    let isMounted = true;
    fetchKBStatusData().then(data => {
      if (isMounted && data) {
        applyStatusData(data);
      }
    });
    return () => { isMounted = false; };
  }, []);

  const handleBuildKB = async () => {
    if (isBusy) return;
    setIsBuilding(true);
    try {
      await buildKnowledgeBase(embeddingModel);
      const data = await fetchKBStatusData();
      if (data) applyStatusData(data);
      alert('构建成功');
    } catch (error) {
      console.error('Failed to build KB:', error);
      alert('构建失败，请检查后端连接');
    } finally {
      setIsBuilding(false);
    }
  };

  const handleCleanKB = async () => {
    if (isBusy) return;
    setIsCleaning(true);
    try {
      await cleanKnowledgeBase();
      const data = await fetchKBStatusData();
      if (data) applyStatusData(data);
      alert('清理成功');
    } catch (error) {
      console.error('Failed to clean KB:', error);
      alert('清理失败，请检查后端连接');
    } finally {
      setIsCleaning(false);
    }
  };

  const handleSyncKB = async () => {
    if (isBusy) return;
    setIsSyncing(true);
    try {
      await syncKnowledgeBase(embeddingModel);
      const data = await fetchKBStatusData();
      if (data) applyStatusData(data);
      alert('同步成功');
    } catch (error) {
      console.error('Failed to sync KB:', error);
      alert('同步失败，请检查后端连接');
    } finally {
      setIsSyncing(false);
    }
  };

  return (
    <div className="mc-sidebar">
      <h2>⛏️ RAG-MCwiki</h2>
      
      <div className="setting-item">
        <label>模型选择</label>
        <select value={model} onChange={(e) => setModel(e.target.value)} className="mc-input">
          <option value="local">本地 (LM Studio)</option>
          <option value="deepseek">DeepSeek API</option>
        </select>
      </div>
      
      {model === 'deepseek' && (
        <div className="setting-item">
          <label>API Key</label>
          <input 
            type="password" 
            value={apiKey} 
            onChange={(e) => setApiKey(e.target.value)} 
            className="mc-input"
            placeholder="Enter DeepSeek Key"
          />
        </div>
      )}

      <div className="setting-item">
        <label>回答详细度</label>
        <select value={detail} onChange={(e) => setDetail(e.target.value)} className="mc-input">
          <option value="简洁">简洁</option>
          <option value="标准">标准</option>
          <option value="详细">详细</option>
        </select>
      </div>

      <div className="setting-item" style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '10px' }}>
        <label htmlFor="proModeToggle" style={{ cursor: 'pointer' }}>专业模式</label>
        <input 
          id="proModeToggle"
          type="checkbox" 
          checked={proMode} 
          onChange={(e) => setProMode(e.target.checked)} 
          style={{ width: '18px', height: '18px', cursor: 'pointer' }}
        />
      </div>

      <hr style={{ borderColor: '#555', margin: '20px 0' }} />

      <h3>知识库管理</h3>
      <div className="setting-item">
        <label>Embedding 模型</label>
        <select value={embeddingModel} onChange={(e) => setEmbeddingModel(e.target.value)} className="mc-input" disabled={isBusy}>
          <option value="shibing624/text2vec-base-chinese">text2vec-base-chinese</option>
          <option value="Qwen/Qwen3-Embedding-0.6B">Qwen3-Embedding-0.6B</option>
        </select>
      </div>
      
      <div className="setting-item" style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '10px' }}>
        <button className="mc-button" onClick={handleBuildKB} disabled={isBusy}>
          {isBuilding ? '正在构建...' : '构建向量库'}
        </button>
        <button className="mc-button" onClick={handleCleanKB} disabled={isBusy} style={{ backgroundColor: '#aa0000' }}>
          {isCleaning ? '正在清理...' : '清理向量库'}
        </button>
        <button className="mc-button" onClick={handleSyncKB} disabled={isBusy} style={{ backgroundColor: '#00aa00' }}>
          {isSyncing ? '正在同步...' : '一键同步最新知识库'}
        </button>
      </div>

      <div style={{ marginTop: 'auto' }}>
        <p style={{ fontSize: '0.8rem', opacity: 0.8, color: '#ffcc00' }}>
          当前本地知识库版本：{kbStatus?.version || '未知'} ({kbStatus?.status === 'ready' ? '就绪' : kbStatus?.status === 'empty' ? '空' : '未知'})
        </p>
        <p style={{ fontSize: '0.8rem', opacity: 0.6 }}>RAG-MCwiki v2.0</p>
      </div>
    </div>
  );
}
