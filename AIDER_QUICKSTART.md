# Aider AI Coding Assistant - Quick Start Guide

## What is Aider?

Aider is an AI pair programming tool that runs in your terminal and uses local Ollama models. It can:
- Generate new code
- Refactor existing code
- Fix bugs
- Write tests
- Work across multiple files
- Auto-commit changes to git

## Hardware Requirements

✅ **Perfect for your setup!**
- RTX 3090 24GB can run both models simultaneously
- qwen2.5-coder:7b uses ~4.7GB VRAM
- qwen2.5-coder:32b uses ~19GB VRAM
- Both together: ~24GB (perfect fit!)

## Quick Setup

```bash
# 1. Install Aider and pull models (one-time setup)
make setup-aider

# 2. Test the setup
bash scripts/test_aider.sh
```

## Usage

### Start Aider

```bash
# Fast model (7B) - Great for quick edits
make aider

# Powerful model (32B) - Better for complex tasks
make aider-32b
```

### Basic Commands

Once in Aider:

```bash
# Add files to the chat context
/add backend/app/main.py
/add frontend/src/App.js

# Ask Aider to make changes
> Add error handling to the chat endpoint
> Refactor the theme toggle to use a context provider
> Add TypeScript types to the frontend components

# Review changes before committing
/diff

# Undo last change
/undo

# Drop files from context (to save memory)
/drop backend/app/main.py

# See all commands
/help

# Exit
/exit
```

## Example Workflows

### 1. Add a New Feature

```bash
$ make aider
Aider v0.59.0

> Add a new API endpoint to export chat history as JSON. 
  It should be a GET request at /api/chat/export/{session_id}

# Aider will:
# 1. Read your codebase
# 2. Edit backend/app/main.py
# 3. Show you the diff
# 4. Auto-commit the changes
```

### 2. Fix a Bug

```bash
$ make aider

> The theme toggle isn't persisting between page refreshes. 
  Fix it by storing the theme in localStorage.

# Aider will analyze and fix the issue
```

### 3. Refactor Code

```bash
$ make aider-32b  # Use 32B for complex refactoring

> Refactor the RAG pipeline to use async/await instead of 
  synchronous calls. Update all related functions.

# 32B model handles complex multi-file refactoring well
```

## Model Comparison

| Feature | 7B Model | 32B Model |
|---------|----------|-----------|
| Speed | ⚡⚡⚡ Fast | ⚡⚡ Moderate |
| Quality | Good | Excellent |
| Context | 8K tokens | 16K tokens |
| VRAM | ~4.7GB | ~19GB |
| Best For | Quick edits, bugs | Complex refactoring |

## Tips & Tricks

### Optimize Performance

1. **Add only relevant files** - Don't add your entire codebase
2. **Use 7B for iteration** - Quick feedback loop
3. **Use 32B for architecture** - Complex changes need more reasoning
4. **Review before committing** - Use `/diff` to see changes

### Working with Large Projects

```bash
# Add specific files, not directories
/add backend/app/main.py
/add backend/app/rag.py

# Use /ls to see what's in context
/ls

# Drop files you don't need anymore
/drop backend/app/rag.py
```

### Git Integration

Aider auto-commits by default. If you want more control:

```bash
# Disable auto-commits in config
auto-commits: false

# Then manually commit good changes
/commit
```

## Configuration Files

Two config files are provided:

### `.aider.conf.yml` (7B model)
- Fast responses
- Good for 90% of coding tasks
- Auto-commits enabled
- Dark mode by default

### `.aider.32b.conf.yml` (32B model)
- More capable reasoning
- Better for complex refactoring
- Larger context window (16K tokens)
- Uses 7B for quick edits (editor-model)

## Troubleshooting

### "Cannot connect to Ollama"

```bash
# Check if Ollama is running
docker ps | grep ollama

# Restart services
make restart

# Test connection
curl http://localhost:11434/api/tags
```

### "Model not found"

```bash
# Pull the models
make setup-aider

# Or manually
docker exec ollama ollama pull qwen2.5-coder:7b
docker exec ollama ollama pull qwen2.5-coder:32b
```

### Out of Memory

```bash
# Use the 7B model instead of 32B
make aider

# Or reduce context in config
max-tokens: 4096
```

## Advanced Usage

### Custom Model Options

```bash
# Use with custom settings
aider \
  --model ollama/qwen2.5-coder:32b \
  --editor-model ollama/qwen2.5-coder:7b \
  --edit-format diff \
  --dark-mode \
  --auto-commits

# Set Ollama API endpoint
OLLAMA_API_BASE=http://localhost:11434 aider
```

### Read-Only Mode

```bash
# Ask questions without editing files
aider --read backend/app/main.py

> Explain how the RAG pipeline works
```

### Architect Mode

```bash
# Get high-level code suggestions
aider --architect

> Design a caching layer for the vector search
```

## Resources

- [Aider Documentation](https://aider.chat/docs/)
- [Qwen2.5-Coder Model](https://ollama.com/library/qwen2.5-coder)
- [Ollama Documentation](https://ollama.com/docs)

## Getting Help

```bash
# In Aider
/help

# Test setup
bash scripts/test_aider.sh

# Check available models
docker exec ollama ollama list

# View Aider version
aider --version
```

---

**Happy Coding with AI! 🚀**
