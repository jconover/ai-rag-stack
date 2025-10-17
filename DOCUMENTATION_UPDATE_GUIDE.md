# Documentation Update System

## Overview

Your AI RAG Stack now includes an automated documentation update system that keeps your knowledge base fresh with the latest information from all 30+ documentation sources.

## Quick Start

### Manual Update

```bash
# Update all documentation to latest versions
make update-docs
```

This command will:
1. ✅ Pull latest changes from all git repositories
2. ✅ Show summary of which repos were updated
3. ✅ Automatically re-ingest if updates found
4. ✅ Display total documentation size

### Automated Updates (Recommended)

Set up n8n for weekly automated updates:

```bash
# 1. Start n8n
docker compose up -d n8n

# 2. Access n8n at http://localhost:5678
# 3. Import n8n-workflows/weekly-doc-update.json
# 4. Configure Slack/Email notifications
# 5. Activate the workflow
```

See [n8n-workflows/README.md](n8n-workflows/README.md) for detailed setup.

---

## How It Works

### Update Script (`scripts/update_docs.sh`)

The update script intelligently updates all documentation repositories:

**Features:**
- Checks all 30+ documentation sources
- Uses `git pull` for repositories with git history
- Tracks before/after commit hashes
- Reports which repos were updated
- Exit code indicates if updates found (0 = updates, 1 = no updates)

**Categories Updated:**
- DevOps & Infrastructure (5 sources)
- Monitoring & Observability (5 sources)
- Programming Languages (9 sources)
- CI/CD & GitOps (5 sources)
- Cloud Platforms (1 source)
- Automation & Integration (2 sources)

### Makefile Command

```makefile
update-docs:
  bash scripts/update_docs.sh data/docs && \
    make ingest || \
    echo "No updates found"
```

**Smart behavior:**
- Only re-ingests if updates detected
- Saves processing time when docs are current
- Provides clear feedback

---

## Usage Scenarios

### Scenario 1: Weekly Manual Updates

**Best for:** Small teams, learning environments

```bash
# Every Monday morning
make update-docs

# Review what changed
cat data/update-log.json

# Test with a query about new features
```

### Scenario 2: Weekly Automated (Recommended)

**Best for:** Production environments, teams

**Setup:**
1. Import `weekly-doc-update.json` to n8n
2. Configure Slack webhook for notifications
3. Activate workflow
4. Receive weekly summary in Slack

**Benefits:**
- Set and forget
- Team visibility via Slack
- Audit trail in update logs
- Consistent update schedule

### Scenario 3: Nightly Automated

**Best for:** Bleeding-edge projects, high-frequency updates

**Setup:**
1. Import `daily-doc-update.json` to n8n
2. Configure Slack webhook
3. Activate workflow
4. Only notified when updates found (silent mode)

**Benefits:**
- Always up-to-date
- Minimal notification fatigue
- Catches security updates quickly

---

## Update Frequency Comparison

### Weekly (Recommended)

| Pros | Cons |
|------|------|
| ✅ Lower resource usage | ⚠️ Week-old information |
| ✅ Predictable schedule | ⚠️ May miss urgent updates |
| ✅ Review-friendly | |
| ✅ Less notification spam | |

**Use when:**
- Documentation doesn't change frequently
- Resource efficiency is important
- Team prefers scheduled review windows

### Daily (Nightly)

| Pros | Cons |
|------|------|
| ✅ Always current | ⚠️ Higher resource usage |
| ✅ Catches urgent updates | ⚠️ More frequent ingestion |
| ✅ Security patches ASAP | ⚠️ Unpredictable timing |

**Use when:**
- Working on cutting-edge projects
- Security is critical
- Need latest API changes immediately

### Manual (On-Demand)

| Pros | Cons |
|------|------|
| ✅ Full control | ⚠️ Easy to forget |
| ✅ Zero overhead when not needed | ⚠️ Inconsistent schedule |
| ✅ Perfect for testing | ⚠️ Manual effort required |

**Use when:**
- Testing the update system
- Before important demos
- Debugging documentation issues

---

## Output Examples

### Update Found

```bash
$ make update-docs

Updating DevOps documentation in data/docs...
Started at: Sun Oct 17 02:00:00 UTC 2025

═══════════════════════════════════════════
DEVOPS & INFRASTRUCTURE
═══════════════════════════════════════════
Updating Kubernetes...
  ✓ Kubernetes updated (a1b2c3d -> e4f5g6h)
Updating Docker...
  ✓ Docker updated (x9y8z7w -> v6u5t4s)
  → Terraform already up-to-date
  → Ansible already up-to-date
  → Helm already up-to-date

...

═══════════════════════════════════════════
Update Summary
═══════════════════════════════════════════
Completed at: Sun Oct 17 02:05:23 UTC 2025

Repositories updated: 3
Failed updates: 0

Updated repositories:
  • Kubernetes
  • Docker
  • Rust

⚠️  Re-ingestion recommended: make ingest

Total size: 1.2G

📚 Updates detected! Re-ingesting documentation...
Ingesting documentation into vector database...
```

### No Updates Found

```bash
$ make update-docs

Updating DevOps documentation in data/docs...
Started at: Sun Oct 17 02:00:00 UTC 2025

...all repos up-to-date...

═══════════════════════════════════════════
Update Summary
═══════════════════════════════════════════
Completed at: Sun Oct 17 02:01:15 UTC 2025

Repositories updated: 0
Failed updates: 0

No repositories had updates.

Total size: 1.2G

✓ No updates found. Documentation is current.
```

---

## Notifications

### Slack Notification (Updates Found)

```
📚 Documentation Update Complete!

✓ 3 repositories updated:
  • Kubernetes
  • Docker
  • Rust

Re-ingestion completed successfully.
```

### Email Notification

**Subject:** AI RAG Stack - Documentation Update Report

**Body:**
```
📚 Documentation Update Complete!

✓ 3 repositories updated:
  • Kubernetes
  • Docker
  • Rust

Re-ingestion completed successfully.
```

---

## Update Log

All updates are logged to:

```
data/update-log.json
```

**Format:**
```json
[
  {
    "timestamp": "2025-10-17T02:00:00.000Z",
    "updatedCount": 3,
    "updatedRepos": ["Kubernetes", "Docker", "Rust"],
    "success": true
  }
]
```

**View logs:**
```bash
# Pretty print
cat data/update-log.json | python3 -m json.tool

# Last 5 updates
cat data/update-log.json | jq '.[-5:]'

# Count total updates
cat data/update-log.json | jq 'length'
```

---

## Troubleshooting

### Updates Not Detected

**Problem:** Running `make update-docs` shows no updates, but you know there should be.

**Solutions:**
```bash
# 1. Check if repos are valid git repositories
cd data/docs/kubernetes && git status

# 2. Manually pull to see if there's an issue
cd data/docs/kubernetes && git pull

# 3. Check remote connection
cd data/docs/kubernetes && git remote -v

# 4. Re-clone if corrupted
rm -rf data/docs/kubernetes
make download-docs
```

### Re-ingestion Fails

**Problem:** Updates detected but re-ingestion fails.

**Solutions:**
```bash
# 1. Check backend is running
docker ps | grep rag-backend

# 2. Check backend logs
docker logs rag-backend

# 3. Manually run ingestion
docker exec rag-backend python /scripts/ingest_docs.py

# 4. Restart backend
docker compose restart backend
```

### Slow Updates

**Problem:** Update process takes too long (>10 minutes).

**Solutions:**
1. **Use SSD storage** for data/docs directory
2. **Limit concurrent git operations** (edit script to process sequentially)
3. **Schedule during off-peak hours** (2 AM is usually good)
4. **Consider incremental ingestion** (future feature)

### n8n Workflow Not Triggering

**Problem:** Scheduled workflow doesn't run.

**Solutions:**
```bash
# 1. Check n8n is running
docker ps | grep n8n

# 2. Check n8n logs
docker logs n8n

# 3. Verify workflow is activated (toggle in UI)

# 4. Check timezone settings in docker-compose
# Make sure TZ environment variable matches your location

# 5. Test manually by clicking "Execute Workflow"
```

---

## Advanced Usage

### Custom Update Schedule

Edit the cron expression in n8n workflow:

```javascript
// Weekly on Sunday at 2 AM
"0 2 * * 0"

// Bi-weekly (1st and 15th at 2 AM)
"0 2 1,15 * *"

// Weekdays only at 2 AM
"0 2 * * 1-5"

// Every 6 hours
"0 */6 * * *"
```

### Selective Updates

Update only specific categories:

```bash
# Edit update_docs.sh to comment out sections you don't want

# Example: Only update programming languages
# Comment out other sections, run:
bash scripts/update_docs.sh data/docs
```

### Pre/Post Update Hooks

Add custom logic before/after updates:

```bash
# Create wrapper script
cat > scripts/update_with_hooks.sh << 'EOF'
#!/bin/bash

echo "Pre-update hook: Backing up vector DB..."
# Your backup logic here

bash scripts/update_docs.sh "$@"

echo "Post-update hook: Running tests..."
# Your test logic here
EOF

chmod +x scripts/update_with_hooks.sh

# Use in Makefile or n8n
```

---

## Best Practices

### 1. Monitor Update Logs

Review update logs periodically:

```bash
# Weekly review
cat data/update-log.json | jq '.[-4:]' | jq -r '.[] | "\(.timestamp): \(.updatedCount) updates"'
```

### 2. Test Before Automating

Always test manually first:

```bash
# Test update process
make update-docs

# Verify queries still work
# Open http://localhost:3000
# Try some test queries
```

### 3. Set Up Notifications

Even if using manual updates, set up notifications for awareness:

```bash
# Simple: Log to file
make update-docs >> logs/updates.log 2>&1

# Advanced: Send email on completion
make update-docs && echo "Updates complete" | mail -s "RAG Update" you@example.com
```

### 4. Version Control Awareness

Remember that updates modify files in `data/docs/`:

```bash
# Option 1: Exclude from git (already in .gitignore)
# This is recommended - docs are reproducible via download

# Option 2: Track changes
git add data/docs
git commit -m "Updated documentation $(date +%Y-%m-%d)"
```

### 5. Disk Space Management

Monitor disk usage:

```bash
# Check current size
du -sh data/docs

# Expected: ~1.2GB for all sources
# If growing significantly, investigate:
du -sh data/docs/* | sort -h

# Clean up old git history if needed
cd data/docs/kubernetes && git gc --aggressive
```

---

## Future Enhancements

Potential improvements for the update system:

- [ ] **Incremental ingestion** - Only re-ingest changed files
- [ ] **Parallel updates** - Update multiple repos simultaneously
- [ ] **Change detection** - Show diff of what changed
- [ ] **Rollback capability** - Revert to previous version
- [ ] **Update metrics** - Dashboard of update history
- [ ] **Selective re-ingestion** - Only ingest updated repos
- [ ] **Health checks** - Verify documentation quality post-update
- [ ] **Automatic cleanup** - Remove stale documentation

---

## Summary

Your documentation update system provides:

✅ **Manual control** via `make update-docs`
✅ **Automated updates** via n8n workflows (weekly/daily)
✅ **Smart re-ingestion** (only when needed)
✅ **Notifications** (Slack, Email)
✅ **Audit trail** (update logs)
✅ **Flexible scheduling** (weekly, daily, on-demand)

**Recommended Setup:**
1. Start with manual updates to test
2. Deploy weekly n8n workflow for production
3. Configure Slack notifications for team visibility
4. Review logs monthly

---

**Files:**
- `scripts/update_docs.sh` - Update script
- `Makefile` - `update-docs` command
- `n8n-workflows/weekly-doc-update.json` - Weekly automation
- `n8n-workflows/daily-doc-update.json` - Daily automation
- `n8n-workflows/README.md` - Setup instructions

**Commands:**
- `make update-docs` - Manual update
- `make download-docs` - Initial download
- `make ingest` - Manual re-ingestion

**Status:** ✅ Production Ready
**Version:** 1.0.0
**Last Updated:** 2025-10-17
