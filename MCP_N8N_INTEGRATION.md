# MCP and n8n Integration Guide

## Overview

Your AI RAG Stack can be extended with **MCP (Model Context Protocol)** and **n8n** to create powerful automation and integration workflows.

## What is MCP (Model Context Protocol)?

MCP is Anthropic's open protocol that allows AI assistants (like Claude) to connect to external data sources and tools. Think of it as a universal connector for AI.

### MCP Integration Possibilities

#### 1. **Expose Your RAG System via MCP**

You can create an MCP server that exposes your RAG system to Claude desktop or other MCP clients:

```python
# Example: backend/mcp_server.py
from mcp import Server
from your_rag import search_documents

server = Server("ai-rag-stack")

@server.tool()
def search_devops_docs(query: str, top_k: int = 5):
    """Search DevOps and programming documentation"""
    return search_documents(query, top_k)

@server.tool()
def list_available_docs():
    """List all available documentation sources"""
    return {
        "devops": ["kubernetes", "terraform", "docker"],
        "languages": ["python", "go", "rust", "javascript"],
        "monitoring": ["prometheus", "grafana", "elk"],
        "cloud": ["aws", "azure", "gcp"]
    }
```

#### 2. **Connect External Data Sources via MCP**

Create MCP servers to ingest data from various sources:

- **Jira/GitHub Issues** - Query project issues and tickets
- **Confluence/Notion** - Access company documentation
- **Slack/Discord** - Search conversation history
- **Internal Databases** - Query production metrics

#### 3. **Tool Calling**

Enable your RAG system to execute actions:

```python
@server.tool()
def run_kubectl_command(namespace: str, command: str):
    """Execute kubectl commands safely"""
    # Validate and execute
    pass

@server.tool()
def check_service_health(service_name: str):
    """Check service health in K8s"""
    pass
```

## What is n8n?

n8n is a workflow automation tool (like Zapier, but self-hosted and open-source). Perfect for DevOps automation!

### n8n Integration Possibilities

#### 1. **Automated Documentation Updates**

```yaml
Workflow: Auto-Update Docs
Trigger: GitHub webhook (new commit)
Actions:
  1. Detect changes in docs repos
  2. Pull updated documentation
  3. Re-ingest into vector database
  4. Send Slack notification
```

#### 2. **AI-Powered Incident Response**

```yaml
Workflow: Incident Assistant
Trigger: PagerDuty alert
Actions:
  1. Parse alert details
  2. Query RAG system for runbooks
  3. Get AI recommendations
  4. Post to Slack with action items
  5. Create Jira ticket
```

#### 3. **Automated Q&A Bot**

```yaml
Workflow: Slack Q&A Bot
Trigger: Slack message in #devops
Actions:
  1. Extract question
  2. Query RAG system
  3. Format response with sources
  4. Reply in Slack thread
  5. Log interaction for analytics
```

#### 4. **CI/CD Documentation Helper**

```yaml
Workflow: Pipeline Documentation
Trigger: CI/CD pipeline fails
Actions:
  1. Extract error message
  2. Query RAG for similar issues
  3. Get AI-generated debugging steps
  4. Comment on PR/commit
  5. Update runbook if new issue
```

#### 5. **Multi-Cloud Cost Monitoring**

```yaml
Workflow: Cost Alerts with Context
Trigger: CloudWatch/GCP alert (high cost)
Actions:
  1. Get resource details
  2. Query RAG for optimization docs
  3. Generate cost-saving recommendations
  4. Create ticket with action plan
  5. Schedule review meeting
```

## Architecture Patterns

### Pattern 1: RAG as MCP Server

```
┌─────────────┐
│ Claude      │
│ Desktop     │
└──────┬──────┘
       │ MCP Protocol
       ↓
┌──────────────────┐
│  MCP Server      │
│  (Python)        │
└──────┬───────────┘
       │ REST API
       ↓
┌──────────────────┐
│  Your RAG Stack  │
│  (FastAPI)       │
└──────────────────┘
```

### Pattern 2: n8n Orchestration

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Trigger   │────▶│    n8n      │────▶│  RAG API    │
│ (Webhook)   │     │  Workflow   │     │             │
└─────────────┘     └─────┬───────┘     └─────────────┘
                          │
                          ↓
                    ┌─────────────┐
                    │  Actions    │
                    │ (Slack, etc)│
                    └─────────────┘
```

### Pattern 3: Full Integration

```
┌──────────────┐
│  External    │
│  Sources     │
│ (Jira, etc)  │
└──────┬───────┘
       │ MCP
       ↓
┌──────────────┐     ┌──────────────┐
│   Claude     │────▶│   n8n        │
│   Desktop    │     │   Workflows  │
└──────┬───────┘     └──────┬───────┘
       │ MCP                 │
       ↓                     ↓
┌──────────────────────────────────┐
│        Your RAG Stack            │
│  (Qdrant + Ollama + FastAPI)    │
└──────────────────────────────────┘
```

## Implementation Examples

### 1. Create MCP Server for RAG

**File: `backend/mcp_server.py`**

```python
#!/usr/bin/env python3
from mcp import Server, Tool
import httpx

server = Server("devops-rag-assistant")

@server.tool()
async def search_docs(query: str, category: str = "all"):
    """
    Search DevOps and programming documentation

    Args:
        query: The search query
        category: Filter by category (devops, programming, cloud, all)
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/chat",
            json={
                "message": query,
                "use_rag": True,
                "category": category
            }
        )
        return response.json()

@server.tool()
async def explain_command(command: str, tool: str):
    """
    Explain what a specific command does

    Args:
        command: The command to explain (e.g., "kubectl get pods")
        tool: The tool (kubectl, terraform, docker, etc.)
    """
    query = f"Explain this {tool} command: {command}"
    return await search_docs(query, "devops")

if __name__ == "__main__":
    server.run()
```

### 2. n8n Workflow: Slack Bot

**Export this as JSON in n8n:**

```json
{
  "name": "DevOps RAG Slack Bot",
  "nodes": [
    {
      "name": "Slack Trigger",
      "type": "n8n-nodes-base.slackTrigger",
      "parameters": {
        "channel": "#devops-questions"
      }
    },
    {
      "name": "Query RAG",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "http://localhost:8000/api/chat",
        "method": "POST",
        "body": {
          "message": "={{$json[\"text\"]}}",
          "use_rag": true
        }
      }
    },
    {
      "name": "Format Response",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "const response = $input.item.json.response;\nconst sources = $input.item.json.sources;\nreturn [{\n  json: {\n    text: `📚 *Answer:*\\n${response}\\n\\n*Sources:* ${sources.length} documents`\n  }\n}];"
      }
    },
    {
      "name": "Reply Slack",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "={{$json[\"channel\"]}}",
        "text": "={{$json[\"text\"]}}"
      }
    }
  ]
}
```

### 3. n8n Workflow: Auto-Update Docs

```json
{
  "name": "Auto Update Documentation",
  "nodes": [
    {
      "name": "Cron Trigger",
      "type": "n8n-nodes-base.cron",
      "parameters": {
        "triggerTimes": {
          "item": [
            {
              "mode": "everyDay",
              "hour": 2
            }
          ]
        }
      }
    },
    {
      "name": "Pull Documentation",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": {
        "command": "cd /path/to/ai-rag-stack && make download-docs"
      }
    },
    {
      "name": "Re-ingest Docs",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": {
        "command": "cd /path/to/ai-rag-stack && make ingest"
      }
    },
    {
      "name": "Notify Slack",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#devops",
        "text": "📚 Documentation updated successfully!"
      }
    }
  ]
}
```

## Getting Started

### Option 1: Add MCP Server

```bash
# 1. Install MCP SDK
pip install mcp

# 2. Create MCP server (see examples above)
nano backend/mcp_server.py

# 3. Run MCP server
python backend/mcp_server.py

# 4. Configure Claude Desktop to use it
# Add to ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "devops-rag": {
      "command": "python",
      "args": ["/path/to/backend/mcp_server.py"]
    }
  }
}
```

### Option 2: Deploy n8n

```bash
# 1. Add n8n to docker-compose.yml
services:
  n8n:
    image: n8nio/n8n:latest
    container_name: n8n
    ports:
      - "5678:5678"
    volumes:
      - n8n_data:/home/node/.n8n
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=changeme
    restart: unless-stopped

# 2. Start n8n
docker compose up -d n8n

# 3. Access n8n
open http://localhost:5678

# 4. Import workflows from examples above
```

## Use Cases

### For DevOps Teams

1. **Incident Response Assistant**
   - Auto-query runbooks during incidents
   - Get AI-powered troubleshooting steps
   - Update documentation based on resolutions

2. **Onboarding Helper**
   - New team members ask questions in Slack
   - Automatic responses from documentation
   - Track common questions for FAQ

3. **Configuration Management**
   - Validate YAML/JSON configs
   - Get best practices for Terraform/K8s
   - Auto-generate documentation

### For Developers

1. **Code Review Assistant**
   - Query language-specific best practices
   - Get security recommendations
   - Find similar code patterns

2. **Learning Assistant**
   - Ask "How do I X in Go/Python/Rust?"
   - Get examples from official docs
   - Compare approaches across languages

3. **CI/CD Helper**
   - Auto-comment on failed pipelines
   - Suggest fixes based on errors
   - Update pipeline docs

## Security Considerations

### MCP Security

1. **Authentication**: Always use authentication for MCP servers
2. **Rate Limiting**: Implement rate limits on tool calls
3. **Input Validation**: Sanitize all inputs before processing
4. **Audit Logging**: Log all MCP interactions

### n8n Security

1. **Network Isolation**: Run n8n in private network
2. **Credentials**: Use n8n's credential management
3. **HTTPS**: Always use TLS for webhooks
4. **Access Control**: Limit who can create workflows

## Resources

- **MCP Specification**: https://github.com/anthropics/anthropic-mcp
- **n8n Documentation**: https://docs.n8n.io
- **n8n Templates**: https://n8n.io/workflows
- **Your RAG API**: http://localhost:8000/docs

## Next Steps

1. **Choose your integration pattern** (MCP, n8n, or both)
2. **Implement a proof-of-concept** with one workflow
3. **Test with real scenarios** from your team
4. **Iterate and expand** based on feedback
5. **Share workflows** with your team

---

**Pro Tip**: Start with simple n8n workflows (like Slack bot) before adding MCP complexity. Once n8n workflows are stable, add MCP for richer AI interactions.
