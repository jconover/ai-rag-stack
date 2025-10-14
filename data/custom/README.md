# Custom Documentation

Place your custom documentation files here to be indexed by the RAG system.

## Supported Formats

- Markdown (`.md`)
- Plain text (`.txt`)

## Examples

```bash
# Add your company's internal docs
cp ~/my-company-docs/*.md .

# Add your personal notes
cp ~/devops-notes/*.txt .

# Add specific tool documentation
wget https://example.com/tool-docs.md
```

## Organizing Files

You can create subdirectories:

```
custom/
├── kubernetes/
│   ├── internal-k8s-setup.md
│   └── troubleshooting-guide.md
├── terraform/
│   └── module-standards.md
└── runbooks/
    ├── incident-response.md
    └── deployment-procedures.md
```

## Re-indexing

After adding files, re-run the ingestion:

```bash
make ingest
```

Or using Docker directly:

```bash
docker exec backend python /app/../scripts/ingest_docs.py
```

## Tips

- Use descriptive filenames
- Keep files focused on specific topics
- Include metadata in headers (title, date, author)
- Use code blocks with language tags for syntax highlighting
