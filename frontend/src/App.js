import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('llama3.1:8b');
  const [sessionId, setSessionId] = useState('');
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [theme, setTheme] = useState(() => {
    // Load theme from localStorage or default to 'dark'
    return localStorage.getItem('theme') || 'dark';
  });
  const [useStreaming, setUseStreaming] = useState(() => {
    return localStorage.getItem('useStreaming') !== 'false';
  });
  const [templates, setTemplates] = useState([]);
  const [showTemplates, setShowTemplates] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Generate session ID
    setSessionId(generateSessionId());

    // Fetch available models
    fetchModels();

    // Fetch stats
    fetchStats();

    // Check health
    fetchHealth();

    // Fetch templates
    fetchTemplates();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Save theme to localStorage when it changes
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    // Save streaming preference
    localStorage.setItem('useStreaming', useStreaming);
  }, [useStreaming]);

  const generateSessionId = () => {
    return 'session_' + Math.random().toString(36).substr(2, 9);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/models`);
      setModels(response.data.models);
      if (response.data.models.length > 0) {
        setSelectedModel(response.data.models[0].name);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/health`);
      setHealth(response.data);
    } catch (error) {
      console.error('Error fetching health:', error);
    }
  };

  const fetchTemplates = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/templates`);
      setTemplates(response.data.templates);
    } catch (error) {
      console.error('Error fetching templates:', error);
    }
  };

  const handleTemplateSelect = (template) => {
    setInput(template.prompt);
    setShowTemplates(false);
  };

  const handleFileUpload = async (files) => {
    if (files.length === 0) return;

    setUploadingFiles(true);
    setUploadProgress('Uploading files...');

    const formData = new FormData();
    for (let file of files) {
      formData.append('files', file);
    }

    try {
      const response = await axios.post(`${API_URL}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setUploadProgress(
        `Successfully uploaded ${response.data.uploaded} file(s). ${
          response.data.ingestion?.status === 'success'
            ? 'Documents ingested into vector database.'
            : 'Ingestion pending.'
        }`
      );

      // Refresh stats
      setTimeout(() => {
        fetchStats();
        setUploadProgress('');
        setShowUpload(false);
      }, 3000);
    } catch (error) {
      setUploadProgress('Upload failed: ' + error.message);
    } finally {
      setUploadingFiles(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(generateSessionId());
    setInput('');
  };

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'dark' ? 'catppuccin' : 'dark');
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input };
    const currentInput = input;
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    // Use streaming if enabled
    if (useStreaming) {
      sendStreamingMessage(currentInput);
    } else {
      sendNonStreamingMessage(currentInput);
    }
  };

  const sendNonStreamingMessage = async (message) => {
    try {
      const response = await axios.post(`${API_URL}/api/chat`, {
        message: message,
        model: selectedModel,
        session_id: sessionId,
        temperature: 0.7,
        use_rag: true,
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.response,
        sources: response.data.sources,
        context_used: response.data.context_used,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'error',
        content: 'Failed to get response. Please check if all services are running.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const sendStreamingMessage = async (message) => {
    let fullResponse = '';
    let metadata = null;
    const messageIndex = messages.length + 1; // +1 for user message already added

    try {
      const response = await fetch(`${API_URL}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          model: selectedModel,
          session_id: sessionId,
          temperature: 0.7,
          use_rag: true,
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'metadata') {
                metadata = data;
              } else if (data.type === 'content') {
                fullResponse += data.content;
                // Update the message in real-time
                setMessages((prev) => {
                  const newMessages = [...prev];
                  if (newMessages[messageIndex]?.role === 'assistant') {
                    newMessages[messageIndex].content = fullResponse;
                  } else {
                    newMessages.push({
                      role: 'assistant',
                      content: fullResponse,
                      sources: metadata?.sources,
                      context_used: metadata?.context_used,
                    });
                  }
                  return newMessages;
                });
              } else if (data.type === 'error') {
                throw new Error(data.error);
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error with streaming:', error);
      const errorMessage = {
        role: 'error',
        content: 'Streaming failed. ' + error.message,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App" data-theme={theme}>
      <header className="App-header">
        <h1>DevOps AI Assistant</h1>
        <div className="header-info">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="model-select"
          >
            {models.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </select>
          {stats && (
            <span className="stats">
              {stats.vectors_count} docs indexed
            </span>
          )}
          {health && (
            <span className={`health-badge ${health.status}`}>
              {health.status}
            </span>
          )}
          <button
            onClick={() => setUseStreaming(!useStreaming)}
            className="streaming-toggle"
            title={`Streaming: ${useStreaming ? 'ON' : 'OFF'}`}
          >
            {useStreaming ? '⚡ Stream' : '📦 Batch'}
          </button>
          <button
            onClick={() => setShowTemplates(!showTemplates)}
            className="templates-button"
            title="Quick prompts"
          >
            📝 Templates
          </button>
          <button
            onClick={() => setShowUpload(!showUpload)}
            className="upload-button"
            title="Upload custom docs"
          >
            📤 Upload
          </button>
          <button
            onClick={toggleTheme}
            className="theme-toggle"
            title={`Switch to ${theme === 'dark' ? 'Catppuccin' : 'Dark'} theme`}
          >
            {theme === 'dark' ? '🎨' : '🌙'}
          </button>
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              className="clear-button"
              title="Start new chat"
            >
              New Chat
            </button>
          )}
        </div>
      </header>

      <div className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <h2>Welcome to DevOps AI Assistant</h2>
              <p>Ask me anything about Kubernetes, Terraform, Docker, Ansible, and more!</p>
              <div className="example-questions">
                <button onClick={() => setInput("How do I create a Kubernetes deployment?")}>
                  How do I create a Kubernetes deployment?
                </button>
                <button onClick={() => setInput("Explain Terraform state management")}>
                  Explain Terraform state management
                </button>
                <button onClick={() => setInput("What's the difference between CMD and ENTRYPOINT in Docker?")}>
                  Docker CMD vs ENTRYPOINT
                </button>
              </div>
            </div>
          )}
          
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.role === 'user' ? (
                  <p>{msg.content}</p>
                ) : msg.role === 'error' ? (
                  <p className="error-text">{msg.content}</p>
                ) : (
                  <>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content}
                    </ReactMarkdown>
                    {msg.context_used && msg.sources && (
                      <details className="sources">
                        <summary>Sources ({msg.sources.length})</summary>
                        <ul>
                          {msg.sources.map((source, i) => (
                            <li key={i}>
                              <strong>{source.source_type}</strong>: {source.source}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="loading-dots">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={sendMessage} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about DevOps tools and practices..."
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </div>

      {/* Templates Modal */}
      {showTemplates && (
        <div className="modal-overlay" onClick={() => setShowTemplates(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Prompt Templates</h2>
              <button onClick={() => setShowTemplates(false)} className="close-button">
                ✕
              </button>
            </div>
            <div className="templates-grid">
              {templates.map((template) => (
                <div
                  key={template.id}
                  className="template-card"
                  onClick={() => handleTemplateSelect(template)}
                >
                  <div className="template-category">{template.category}</div>
                  <div className="template-title">{template.title}</div>
                  <div className="template-description">{template.description}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {showUpload && (
        <div className="modal-overlay" onClick={() => setShowUpload(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Upload Documentation</h2>
              <button onClick={() => setShowUpload(false)} className="close-button">
                ✕
              </button>
            </div>
            <div className="upload-area">
              <p>Upload custom documentation files (.md, .txt, .markdown)</p>
              <input
                type="file"
                multiple
                accept=".md,.txt,.markdown"
                onChange={(e) => handleFileUpload(Array.from(e.target.files))}
                disabled={uploadingFiles}
                id="file-upload"
              />
              <label htmlFor="file-upload" className="upload-label">
                {uploadingFiles ? 'Uploading...' : 'Choose Files'}
              </label>
              {uploadProgress && (
                <div className={`upload-progress ${uploadProgress.includes('Success') ? 'success' : ''}`}>
                  {uploadProgress}
                </div>
              )}
              <div className="upload-info">
                <p>📚 Files will be automatically indexed into the vector database</p>
                <p>💡 This allows the AI to answer questions about your custom documentation</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
