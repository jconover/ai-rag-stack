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
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/api/chat`, {
        message: input,
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

  return (
    <div className="App">
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
    </div>
  );
}

export default App;
