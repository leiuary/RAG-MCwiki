'use client';

import { useState } from 'react';
import Sidebar from '../components/Sidebar';
import ChatArea from '../components/ChatArea';

export default function Home() {
  const [model, setModel] = useState('local');
  const [detail, setDetail] = useState('标准');
  const [apiKey, setApiKey] = useState('');
  const [proMode, setProMode] = useState(false);

  return (
    <div className="mc-container">
      <Sidebar 
        model={model} setModel={setModel}
        apiKey={apiKey} setApiKey={setApiKey}
        detail={detail} setDetail={setDetail}
        proMode={proMode} setProMode={setProMode}
      />
      <ChatArea 
        model={model}
        apiKey={apiKey}
        detail={detail}
        proMode={proMode}
      />
    </div>
  );
}
