#!/bin/bash

# DevOps Documentation Updater
# This script updates existing documentation repositories with latest changes

set -e

DOCS_DIR="${1:-../data/docs}"
UPDATED_COUNT=0
FAILED_COUNT=0
UPDATED_REPOS=()

echo "Updating DevOps documentation in $DOCS_DIR..."
echo "Started at: $(date)"
echo ""

# Function to update a git repository
update_repo() {
    local repo_name=$1
    local repo_path=$2

    if [ -d "$repo_path/.git" ]; then
        echo "Updating $repo_name..."
        cd "$repo_path"

        # Get current commit hash
        OLD_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

        # Pull latest changes
        if git pull origin $(git rev-parse --abbrev-ref HEAD) 2>/dev/null; then
            NEW_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

            if [ "$OLD_COMMIT" != "$NEW_COMMIT" ]; then
                echo "  ✓ $repo_name updated (${OLD_COMMIT:0:7} -> ${NEW_COMMIT:0:7})"
                UPDATED_COUNT=$((UPDATED_COUNT + 1))
                UPDATED_REPOS+=("$repo_name")
            else
                echo "  → $repo_name already up-to-date"
            fi
        else
            echo "  ✗ Failed to update $repo_name"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi

        cd - > /dev/null
    elif [ -d "$repo_path" ]; then
        echo "  ⚠ $repo_name exists but is not a git repository, skipping..."
    else
        echo "  ⚠ $repo_name not found, skipping... (run 'make download-docs' first)"
    fi
}

# DevOps & Infrastructure
echo "═══════════════════════════════════════════"
echo "DEVOPS & INFRASTRUCTURE"
echo "═══════════════════════════════════════════"
update_repo "Kubernetes" "$DOCS_DIR/kubernetes"
update_repo "Terraform" "$DOCS_DIR/terraform"
update_repo "Docker" "$DOCS_DIR/docker"
update_repo "Ansible" "$DOCS_DIR/ansible"
update_repo "Helm" "$DOCS_DIR/helm"
echo ""

# Monitoring & Observability
echo "═══════════════════════════════════════════"
echo "MONITORING & OBSERVABILITY"
echo "═══════════════════════════════════════════"
update_repo "Prometheus" "$DOCS_DIR/prometheus"
update_repo "Grafana" "$DOCS_DIR/grafana"
update_repo "Elasticsearch" "$DOCS_DIR/elasticsearch"
update_repo "Logstash" "$DOCS_DIR/logstash"
update_repo "Kibana" "$DOCS_DIR/kibana"
echo ""

# Programming Languages
echo "═══════════════════════════════════════════"
echo "PROGRAMMING LANGUAGES"
echo "═══════════════════════════════════════════"
update_repo "Python" "$DOCS_DIR/python"
update_repo "Go" "$DOCS_DIR/go"
update_repo "Rust" "$DOCS_DIR/rust"
update_repo "Rust by Example" "$DOCS_DIR/rust/by-example"
update_repo "Node.js" "$DOCS_DIR/nodejs"
update_repo "JavaScript (MDN)" "$DOCS_DIR/javascript"
update_repo "Bash Handbook" "$DOCS_DIR/bash/handbook"
update_repo "Zsh Manual" "$DOCS_DIR/zsh/manual"
update_repo "Zsh Completions" "$DOCS_DIR/zsh/completions"
echo ""

# CI/CD & GitOps
echo "═══════════════════════════════════════════"
echo "CI/CD & GITOPS"
echo "═══════════════════════════════════════════"
update_repo "Pro Git Book" "$DOCS_DIR/git/progit2"
update_repo "Git Official Docs" "$DOCS_DIR/git/official-docs"
update_repo "Jenkins" "$DOCS_DIR/jenkins"
update_repo "GitHub Actions" "$DOCS_DIR/github-actions"
update_repo "ArgoCD" "$DOCS_DIR/argocd"
echo ""

# Cloud Platforms
echo "═══════════════════════════════════════════"
echo "CLOUD PLATFORMS"
echo "═══════════════════════════════════════════"
update_repo "GCP Python Client" "$DOCS_DIR/gcp/python-client"
echo "  → AWS and Azure docs are not git repositories (external sources)"
echo ""

# Automation & Integration
echo "═══════════════════════════════════════════"
echo "AUTOMATION & INTEGRATION"
echo "═══════════════════════════════════════════"
update_repo "n8n" "$DOCS_DIR/n8n"
update_repo "JSON Schema" "$DOCS_DIR/config-formats/json-schema"
echo ""

# Summary
echo "═══════════════════════════════════════════"
echo "Update Summary"
echo "═══════════════════════════════════════════"
echo "Completed at: $(date)"
echo ""
echo "Repositories updated: $UPDATED_COUNT"
echo "Failed updates: $FAILED_COUNT"
echo ""

if [ ${#UPDATED_REPOS[@]} -gt 0 ]; then
    echo "Updated repositories:"
    for repo in "${UPDATED_REPOS[@]}"; do
        echo "  • $repo"
    done
    echo ""
    echo "⚠️  Re-ingestion recommended: make ingest"
else
    echo "No repositories had updates."
fi

echo ""
echo "Total size: $(du -sh $DOCS_DIR | cut -f1)"
echo ""

# Exit with status
if [ $UPDATED_COUNT -gt 0 ]; then
    exit 0  # Updates available
else
    exit 1  # No updates
fi
