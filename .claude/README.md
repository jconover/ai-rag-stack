# Claude Code Configuration

This directory contains [Claude Code](https://claude.ai/code) configuration for AI-assisted development.

## Agents

The `agents/` directory contains 12 specialized AI/ML subagents for Claude Code:

| Agent | Specialty |
|-------|-----------|
| `ai-engineer` | AI system design, model implementation, production deployment |
| `data-analyst` | Business intelligence, data visualization, statistical analysis |
| `data-engineer` | Data pipelines, ETL/ELT, data infrastructure |
| `data-scientist` | Statistical analysis, ML modeling, data storytelling |
| `database-optimizer` | Query optimization, performance tuning, scalability |
| `llm-architect` | LLM architecture, fine-tuning, production serving |
| `machine-learning-engineer` | Model deployment, serving infrastructure, edge deployment |
| `ml-engineer` | ML model lifecycle, training to serving |
| `mlops-engineer` | ML infrastructure, CI/CD for ML, platform engineering |
| `nlp-engineer` | NLP processing, transformer models, text pipelines |
| `postgres-pro` | PostgreSQL administration, optimization, high availability |
| `prompt-engineer` | Prompt design, optimization, evaluation frameworks |

### Usage

These agents can be invoked via Claude Code's Task tool:

```
Use the Task tool with subagent_type="ai-engineer" to review AI architecture
```

Or run multiple agents in parallel for comprehensive reviews:

```
Launch 12 agents to review this project from different expert perspectives
```

### Example: Parallel Expert Review

```bash
# In Claude Code, you can request:
"Run 12 subagents to review this project and give me recommendations"

# This launches all agents in parallel, each providing domain-specific insights
```

## Credits

These agents are sourced from the **[awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents)** repository, specifically from `categories/05-data-ai`.

Thanks to [VoltAgent](https://github.com/VoltAgent) for curating this excellent collection of Claude Code subagents.

## Files

- `agents/*.md` - Subagent definitions (committed to repo)
- `settings.local.json` - Personal settings (gitignored)
