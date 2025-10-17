# Documentation Guide

## Overview

Your AI RAG Stack now includes comprehensive documentation for both DevOps tools and programming languages, providing a full-stack knowledge base for infrastructure and development questions.

## Available Documentation

### DevOps & Infrastructure
| Tool | Source | Content |
|------|--------|---------|
| Kubernetes | [kubernetes/website](https://github.com/kubernetes/website) | Concepts, tutorials, reference |
| Terraform | [hashicorp/terraform-docs-common](https://github.com/hashicorp/terraform-docs-common) | Configuration, providers, modules |
| Docker | [docker/docs](https://github.com/docker/docs) | Engine, Compose, Swarm |
| Ansible | [ansible/ansible-documentation](https://github.com/ansible/ansible-documentation) | Playbooks, roles, modules |
| Prometheus | [prometheus/docs](https://github.com/prometheus/docs) | Metrics, queries, alerting |

### Programming Languages
| Language | Source | Content |
|----------|--------|---------|
| Python | [python/cpython/Doc](https://github.com/python/cpython) | Standard library, language reference |
| Go | [golang/go/doc](https://github.com/golang/go) | Language spec, tutorials, packages |
| Bash | GNU Manual + [bash-handbook](https://github.com/denysdovhan/bash-handbook) | Built-ins, scripting, best practices |
| Zsh | [zsh-users/zsh](https://github.com/zsh-users/zsh) + completions | Manual, completions, hooks |

### Cloud Platforms
| Platform | Content |
|----------|---------|
| AWS | EC2, S3, Lambda, ECS, VPC, IAM |
| Azure | AKS, DevOps, Container Instances |
| GitLab CI/CD | Pipeline configuration, runners |

## Managing Documentation

### Download Documentation

```bash
# Download all documentation
make download-docs

# Or use the script directly
bash scripts/download_docs.sh data/docs

# Download specific categories by editing download_docs.sh
```

### Ingest into Vector Database

```bash
# Ingest all documentation
make ingest

# Or manually
docker exec rag-backend python /scripts/ingest_docs.py

# Check ingestion status
make stats
```

### Update Documentation

```bash
# Remove existing docs
rm -rf data/docs/*

# Re-download
make download-docs

# Re-ingest
make ingest
```

## Adding New Documentation Sources

### Method 1: Edit download_docs.sh

Add a new section to `scripts/download_docs.sh`:

```bash
# Rust Documentation
echo "Downloading Rust docs..."
if [ ! -d "$DOCS_DIR/rust" ]; then
    git clone --depth 1 https://github.com/rust-lang/rust.git "$DOCS_DIR/rust-src"
    if [ -d "$DOCS_DIR/rust-src/src/doc" ]; then
        mv "$DOCS_DIR/rust-src/src/doc" "$DOCS_DIR/rust"
        rm -rf "$DOCS_DIR/rust-src"
    fi
else
    echo "Rust docs already exist, skipping..."
fi
```

### Method 2: Custom Documentation

Add your own markdown/text files:

```bash
# Create custom directory
mkdir -p data/custom/my-project

# Add your docs
cp -r ~/my-project/docs/* data/custom/my-project/

# Ingest
make ingest
```

### Method 3: Manual Git Clone

```bash
# Clone directly to data/docs
cd data/docs
git clone --depth 1 https://github.com/user/project-docs.git project-name

# Ingest
make ingest
```

## Query Examples

### DevOps Queries

```
Infrastructure:
  - "How do I create a Kubernetes deployment with 3 replicas?"
  - "Show me a Terraform module for an AWS VPC"
  - "What's the difference between Docker CMD and ENTRYPOINT?"
  - "How do I use Ansible vault for secrets?"

Monitoring:
  - "Create a Prometheus query for CPU usage"
  - "How do I set up alerts in Prometheus?"
```

### Programming Queries

```
Python:
  - "Explain Python context managers"
  - "How do I use async/await in Python?"
  - "What's the difference between @staticmethod and @classmethod?"
  - "Show me how to use Python dataclasses"

Go:
  - "How do I create a REST API in Go?"
  - "Explain Go interfaces with examples"
  - "What are channels and goroutines?"
  - "How does defer work in Go?"

Bash/Zsh:
  - "How do I parse command line arguments in Bash?"
  - "Show me Bash array manipulation"
  - "How do I create Zsh completions?"
  - "What's the difference between Bash and Zsh arrays?"
```

### Cross-Domain Queries

```
Integration:
  - "How do I deploy a Python Flask app on Kubernetes?"
  - "Show me a CI/CD pipeline for a Go application"
  - "How do I use Terraform to provision infrastructure for a Python app?"
  - "Create a Docker multi-stage build for a Go binary"

Comparison:
  - "Compare error handling in Python, Go, and Bash"
  - "What are the differences between Kubernetes and Docker Swarm?"
  - "Compare async patterns in Python and Go"
```

## Documentation Statistics

Check your vector database stats:

```bash
# Get current stats
make stats

# Example output:
{
  "vectors_count": 15234,
  "indexed_points": 15234,
  "segments_count": 5,
  "disk_size_bytes": 45234567
}
```

## Storage Requirements

| Category | Approximate Size |
|----------|-----------------|
| Kubernetes | ~350 MB |
| Terraform | ~50 MB |
| Docker | ~80 MB |
| Ansible | ~20 MB |
| Python | ~16 MB |
| Go | ~1 MB |
| Bash | ~2 MB |
| Zsh | ~5 MB |
| **Total** | **~524 MB** |

Vector database storage (Qdrant):
- Raw docs: ~524 MB
- Vectors: ~100-200 MB (depends on embedding model)
- Total: ~700 MB

## Performance Optimization

### Selective Ingestion

Edit `scripts/ingest_docs.py` to ingest only specific directories:

```python
# Only ingest Python and Go docs
doc_dirs = [
    'data/docs/python',
    'data/docs/go'
]
```

### Chunk Size Tuning

Adjust in `backend/app/vectorstore.py`:

```python
# Smaller chunks = more precise, more vectors
chunk_size = 500  # Default: 1000

# Larger chunks = more context, fewer vectors
chunk_size = 2000
```

### Top K Results

Adjust in `backend/app/rag.py`:

```python
# More results = more context, slower
top_k = 10  # Default: 5

# Fewer results = faster, less context
top_k = 3
```

## Troubleshooting

### Documentation Not Found

```bash
# Check if docs directory exists
ls -la data/docs/

# Re-download
make download-docs
```

### Ingestion Failed

```bash
# Check backend logs
make logs-backend

# Try manual ingestion
docker exec -it rag-backend python /scripts/ingest_docs.py

# Check Qdrant connection
curl http://localhost:6333/collections
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean up old docs
rm -rf data/docs/*

# Download only essential docs
# Edit scripts/download_docs.sh to comment out large repos
```

### Queries Return No Results

```bash
# Check vector count
make stats

# Verify documents are indexed
curl http://localhost:6333/collections/documents/points

# Re-ingest if needed
make ingest
```

## Maintenance

### Regular Updates

```bash
# Monthly update routine
cd data/docs

# Update each repo
for dir in */; do
    if [ -d "$dir/.git" ]; then
        echo "Updating $dir"
        cd "$dir"
        git pull
        cd ..
    fi
done

# Re-ingest updated docs
make ingest
```

### Backup Vector Database

```bash
# Backup Qdrant data
docker cp qdrant:/qdrant/storage ./backup-qdrant-$(date +%Y%m%d)

# Restore
docker cp ./backup-qdrant-20231215 qdrant:/qdrant/storage
docker restart qdrant
```

## Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama Models](https://ollama.com/library)

## Adding More Languages

Want to add more? Here are some suggestions:

```bash
# Rust
git clone --depth 1 https://github.com/rust-lang/rust.git

# JavaScript/Node.js
git clone --depth 1 https://github.com/nodejs/node.git

# TypeScript
git clone --depth 1 https://github.com/microsoft/TypeScript-Website.git

# Ruby
git clone --depth 1 https://github.com/ruby/ruby.git

# Java
git clone --depth 1 https://github.com/openjdk/jdk.git
```

Then add them to `scripts/download_docs.sh` and `make ingest`!
