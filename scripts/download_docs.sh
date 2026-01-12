#!/bin/bash

# DevOps Documentation Downloader
# This script clones and organizes documentation from various DevOps tools
# Supports parallel downloads with configurable concurrency

# Configuration
DOCS_DIR="${1:-../data/docs}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"  # Default 4 concurrent downloads
MAX_RETRIES="${MAX_RETRIES:-3}"      # Retry failed downloads up to 3 times
RETRY_DELAY="${RETRY_DELAY:-5}"      # Seconds to wait between retries

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create docs directory
mkdir -p "$DOCS_DIR"

# Temporary files for tracking progress
PROGRESS_DIR=$(mktemp -d)
FAILED_FILE="$PROGRESS_DIR/failed.log"
SUCCESS_FILE="$PROGRESS_DIR/success.log"
SKIPPED_FILE="$PROGRESS_DIR/skipped.log"
touch "$FAILED_FILE" "$SUCCESS_FILE" "$SKIPPED_FILE"

# Cleanup on exit
cleanup() {
    rm -rf "$PROGRESS_DIR"
}
trap cleanup EXIT

# Log functions with timestamps
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%H:%M:%S') $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $1"
}

# Download function with retry logic
# Usage: download_repo "name" "url" "target_dir" "extract_subdir" "temp_suffix"
download_repo() {
    local name="$1"
    local url="$2"
    local target_dir="$3"
    local extract_subdir="$4"  # Optional: extract only this subdir
    local temp_suffix="$5"     # Optional: temp directory suffix

    local full_target="$DOCS_DIR/$target_dir"
    local temp_dir=""

    # Check if already exists
    if [ -d "$full_target" ]; then
        log_info "[$name] Already exists, skipping..."
        echo "$name" >> "$SKIPPED_FILE"
        return 0
    fi

    # Determine if we need a temp directory
    if [ -n "$extract_subdir" ]; then
        temp_dir="$DOCS_DIR/${temp_suffix:-${target_dir}-src}"
        rm -rf "$temp_dir" 2>/dev/null || true
    fi

    local attempt=0
    local success=false

    while [ $attempt -lt $MAX_RETRIES ] && [ "$success" = false ]; do
        attempt=$((attempt + 1))

        if [ $attempt -gt 1 ]; then
            log_warning "[$name] Retry $attempt of $MAX_RETRIES..."
            sleep "$RETRY_DELAY"
        fi

        log_info "[$name] Downloading from $url..."

        if [ -n "$extract_subdir" ]; then
            # Clone to temp, extract subdir
            if git clone --depth 1 --quiet "$url" "$temp_dir" 2>/dev/null; then
                if [ -d "$temp_dir/$extract_subdir" ]; then
                    mv "$temp_dir/$extract_subdir" "$full_target"
                    rm -rf "$temp_dir"
                    success=true
                else
                    log_warning "[$name] Subdir '$extract_subdir' not found in repo"
                    rm -rf "$temp_dir"
                fi
            fi
        else
            # Clone directly to target
            if git clone --depth 1 --quiet "$url" "$full_target" 2>/dev/null; then
                success=true
            fi
        fi
    done

    if [ "$success" = true ]; then
        log_success "[$name] Downloaded successfully"
        echo "$name" >> "$SUCCESS_FILE"
        return 0
    else
        log_error "[$name] Failed after $MAX_RETRIES attempts"
        echo "$name" >> "$FAILED_FILE"
        return 1
    fi
}

# Download with curl (for non-git sources)
download_curl() {
    local name="$1"
    local url="$2"
    local target_file="$3"

    local full_target="$DOCS_DIR/$target_file"
    local target_dir=$(dirname "$full_target")

    # Check if already exists
    if [ -f "$full_target" ]; then
        log_info "[$name] Already exists, skipping..."
        echo "$name" >> "$SKIPPED_FILE"
        return 0
    fi

    mkdir -p "$target_dir"

    local attempt=0
    local success=false

    while [ $attempt -lt $MAX_RETRIES ] && [ "$success" = false ]; do
        attempt=$((attempt + 1))

        if [ $attempt -gt 1 ]; then
            log_warning "[$name] Retry $attempt of $MAX_RETRIES..."
            sleep "$RETRY_DELAY"
        fi

        log_info "[$name] Downloading from $url..."

        if curl -sL -o "$full_target" "$url" 2>/dev/null; then
            if [ -s "$full_target" ]; then
                success=true
            else
                rm -f "$full_target"
            fi
        fi
    done

    if [ "$success" = true ]; then
        log_success "[$name] Downloaded successfully"
        echo "$name" >> "$SUCCESS_FILE"
        return 0
    else
        log_error "[$name] Failed after $MAX_RETRIES attempts"
        echo "$name" >> "$FAILED_FILE"
        return 1
    fi
}

# Complex download task (multi-step)
download_complex() {
    local name="$1"
    shift
    # Remaining args are the script to execute

    log_info "[$name] Starting complex download..."

    if "$@"; then
        log_success "[$name] Downloaded successfully"
        echo "$name" >> "$SUCCESS_FILE"
        return 0
    else
        log_error "[$name] Failed"
        echo "$name" >> "$FAILED_FILE"
        return 1
    fi
}

# Export functions and variables for parallel execution
export -f download_repo download_curl download_complex log_info log_success log_warning log_error
export DOCS_DIR PROGRESS_DIR FAILED_FILE SUCCESS_FILE SKIPPED_FILE MAX_RETRIES RETRY_DELAY
export RED GREEN YELLOW BLUE NC

# Define all download tasks
# Format: "function|name|arg1|arg2|..."
DOWNLOAD_TASKS=(
    # Simple git clones
    "download_repo|Kubernetes|https://github.com/kubernetes/website.git|kubernetes"
    "download_repo|Terraform|https://github.com/hashicorp/terraform-docs-common.git|terraform"
    "download_repo|Docker|https://github.com/docker/docs.git|docker"
    "download_repo|Ansible|https://github.com/ansible/ansible-documentation.git|ansible"
    "download_repo|Prometheus|https://github.com/prometheus/docs.git|prometheus"
    "download_repo|n8n|https://github.com/n8n-io/n8n-docs.git|n8n"

    # Repos with subdir extraction
    "download_repo|Python|https://github.com/python/cpython.git|python|Doc|python-src"
    "download_repo|Go|https://github.com/golang/go.git|go|doc|go-src"
    "download_repo|Grafana|https://github.com/grafana/grafana.git|grafana|docs|grafana-src"
    "download_repo|Elasticsearch|https://github.com/elastic/elasticsearch.git|elasticsearch|docs|es-src"
    "download_repo|Logstash|https://github.com/elastic/logstash.git|logstash|docs|logstash-src"
    "download_repo|Kibana|https://github.com/elastic/kibana.git|kibana|docs|kibana-src"
    "download_repo|Jenkins|https://github.com/jenkins-infra/jenkins.io.git|jenkins|content|jenkins-src"
    "download_repo|ArgoCD|https://github.com/argoproj/argo-cd.git|argocd|docs|argocd-src"
    "download_repo|Helm|https://github.com/helm/helm.git|helm|docs|helm-src"
    "download_repo|GCP|https://github.com/googleapis/google-cloud-python.git|gcp/python-client|docs|gcp-tmp"
    "download_repo|Rust-Base|https://github.com/rust-lang/rust.git|rust|src/doc|rust-src"
    "download_repo|NodeJS|https://github.com/nodejs/node.git|nodejs|doc|nodejs-src"
    "download_repo|Zsh-Base|https://github.com/zsh-users/zsh.git|zsh/manual|Doc|zsh-src"
    "download_repo|Git-Official|https://github.com/git/git.git|git/official-docs|Documentation|git-src"
    "download_repo|K8s-AI-ClusterAPI|https://github.com/kubernetes-sigs/cluster-api.git|kubernetes-ai|docs|k8s-ai-src"
    "download_repo|GitHub-Actions|https://github.com/github/docs.git|github-actions/docs|content/actions|gh-docs-tmp"
    "download_repo|JavaScript-MDN|https://github.com/mdn/content.git|javascript/mdn|files/en-us/web/javascript|mdn-js-tmp"

    # Additional resources (direct clones)
    "download_repo|Rust-ByExample|https://github.com/rust-lang/rust-by-example.git|rust/by-example"
    "download_repo|Git-ProGit|https://github.com/progit/progit2.git|git/progit2"
    "download_repo|Bash-Handbook|https://github.com/denysdovhan/bash-handbook.git|bash/handbook"
    "download_repo|Zsh-Completions|https://github.com/zsh-users/zsh-completions.git|zsh/completions"
    "download_repo|JSON-Schema|https://github.com/json-schema-org/json-schema-spec.git|config-formats/json-schema"
    "download_repo|Kubeflow|https://github.com/kubeflow/website.git|kubernetes-ai/kubeflow/docs|content|kubeflow-tmp"

    # Curl downloads
    "download_curl|Bash-Manual|https://www.gnu.org/software/bash/manual/bash.html|bash/bash.html"
    "download_curl|YAML-Spec|https://yaml.org/spec/1.2.2/|config-formats/yaml-spec.md"
)

# Process a single task
process_task() {
    local task="$1"
    IFS='|' read -ra ARGS <<< "$task"
    local func="${ARGS[0]}"

    case "$func" in
        download_repo)
            download_repo "${ARGS[1]}" "${ARGS[2]}" "${ARGS[3]}" "${ARGS[4]:-}" "${ARGS[5]:-}"
            ;;
        download_curl)
            download_curl "${ARGS[1]}" "${ARGS[2]}" "${ARGS[3]}"
            ;;
    esac
}

export -f process_task

# Main execution
echo ""
echo "========================================================"
echo "  DevOps Documentation Downloader (Parallel Edition)"
echo "========================================================"
echo ""
log_info "Target directory: $DOCS_DIR"
log_info "Parallel jobs: $PARALLEL_JOBS"
log_info "Max retries: $MAX_RETRIES"
log_info "Total sources: ${#DOWNLOAD_TASKS[@]}"
echo ""
echo "--------------------------------------------------------"
echo ""

START_TIME=$(date +%s)

# Check if GNU parallel is available, fall back to xargs
if command -v parallel &> /dev/null; then
    log_info "Using GNU parallel for concurrent downloads..."
    printf '%s\n' "${DOWNLOAD_TASKS[@]}" | parallel -j "$PARALLEL_JOBS" --halt never process_task {}
else
    log_info "Using xargs for concurrent downloads (install GNU parallel for better progress)..."
    printf '%s\n' "${DOWNLOAD_TASKS[@]}" | xargs -P "$PARALLEL_JOBS" -I {} bash -c 'process_task "$@"' _ {}
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Count results
SUCCESS_COUNT=$(wc -l < "$SUCCESS_FILE" 2>/dev/null || echo 0)
FAILED_COUNT=$(wc -l < "$FAILED_FILE" 2>/dev/null || echo 0)
SKIPPED_COUNT=$(wc -l < "$SKIPPED_FILE" 2>/dev/null || echo 0)

echo ""
echo "========================================================"
echo "  Download Summary"
echo "========================================================"
echo ""
echo -e "${GREEN}Successful:${NC} $SUCCESS_COUNT"
echo -e "${YELLOW}Skipped:${NC}    $SKIPPED_COUNT (already existed)"
echo -e "${RED}Failed:${NC}     $FAILED_COUNT"
echo ""
echo "Duration: ${DURATION}s"
echo "Total size: $(du -sh "$DOCS_DIR" 2>/dev/null | cut -f1)"
echo ""

# Show failed downloads if any
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo -e "${RED}Failed downloads:${NC}"
    cat "$FAILED_FILE" | while read -r name; do
        echo "  - $name"
    done
    echo ""
fi

echo "Documentation categories:"
echo ""
echo "DEVOPS & INFRASTRUCTURE:"
[ -d "$DOCS_DIR/kubernetes" ] && echo -e "  ${GREEN}[OK]${NC} Kubernetes" || echo -e "  ${RED}[--]${NC} Kubernetes"
[ -d "$DOCS_DIR/kubernetes-ai" ] && echo -e "  ${GREEN}[OK]${NC} Kubernetes AI (Cluster API, Kubeflow)" || echo -e "  ${RED}[--]${NC} Kubernetes AI"
[ -d "$DOCS_DIR/terraform" ] && echo -e "  ${GREEN}[OK]${NC} Terraform" || echo -e "  ${RED}[--]${NC} Terraform"
[ -d "$DOCS_DIR/docker" ] && echo -e "  ${GREEN}[OK]${NC} Docker" || echo -e "  ${RED}[--]${NC} Docker"
[ -d "$DOCS_DIR/ansible" ] && echo -e "  ${GREEN}[OK]${NC} Ansible" || echo -e "  ${RED}[--]${NC} Ansible"
[ -d "$DOCS_DIR/helm" ] && echo -e "  ${GREEN}[OK]${NC} Helm" || echo -e "  ${RED}[--]${NC} Helm"
echo ""
echo "MONITORING & LOGGING:"
[ -d "$DOCS_DIR/prometheus" ] && echo -e "  ${GREEN}[OK]${NC} Prometheus" || echo -e "  ${RED}[--]${NC} Prometheus"
[ -d "$DOCS_DIR/grafana" ] && echo -e "  ${GREEN}[OK]${NC} Grafana" || echo -e "  ${RED}[--]${NC} Grafana"
[ -d "$DOCS_DIR/elasticsearch" ] && echo -e "  ${GREEN}[OK]${NC} Elasticsearch" || echo -e "  ${RED}[--]${NC} Elasticsearch"
[ -d "$DOCS_DIR/logstash" ] && echo -e "  ${GREEN}[OK]${NC} Logstash" || echo -e "  ${RED}[--]${NC} Logstash"
[ -d "$DOCS_DIR/kibana" ] && echo -e "  ${GREEN}[OK]${NC} Kibana" || echo -e "  ${RED}[--]${NC} Kibana"
echo ""
echo "PROGRAMMING LANGUAGES:"
[ -d "$DOCS_DIR/python" ] && echo -e "  ${GREEN}[OK]${NC} Python" || echo -e "  ${RED}[--]${NC} Python"
[ -d "$DOCS_DIR/go" ] && echo -e "  ${GREEN}[OK]${NC} Go" || echo -e "  ${RED}[--]${NC} Go"
[ -d "$DOCS_DIR/rust" ] && echo -e "  ${GREEN}[OK]${NC} Rust" || echo -e "  ${RED}[--]${NC} Rust"
[ -d "$DOCS_DIR/nodejs" ] || [ -d "$DOCS_DIR/javascript" ] && echo -e "  ${GREEN}[OK]${NC} JavaScript/Node.js" || echo -e "  ${RED}[--]${NC} JavaScript/Node.js"
[ -d "$DOCS_DIR/bash" ] && echo -e "  ${GREEN}[OK]${NC} Bash" || echo -e "  ${RED}[--]${NC} Bash"
[ -d "$DOCS_DIR/zsh" ] && echo -e "  ${GREEN}[OK]${NC} Zsh" || echo -e "  ${RED}[--]${NC} Zsh"
echo ""
echo "CI/CD & GITOPS:"
[ -d "$DOCS_DIR/git" ] && echo -e "  ${GREEN}[OK]${NC} Git" || echo -e "  ${RED}[--]${NC} Git"
[ -d "$DOCS_DIR/jenkins" ] && echo -e "  ${GREEN}[OK]${NC} Jenkins" || echo -e "  ${RED}[--]${NC} Jenkins"
[ -d "$DOCS_DIR/github-actions" ] && echo -e "  ${GREEN}[OK]${NC} GitHub Actions" || echo -e "  ${RED}[--]${NC} GitHub Actions"
[ -d "$DOCS_DIR/argocd" ] && echo -e "  ${GREEN}[OK]${NC} ArgoCD" || echo -e "  ${RED}[--]${NC} ArgoCD"
echo ""
echo "CLOUD PLATFORMS:"
[ -d "$DOCS_DIR/gcp" ] && echo -e "  ${GREEN}[OK]${NC} GCP" || echo -e "  ${RED}[--]${NC} GCP"
echo ""
echo "AUTOMATION & INTEGRATION:"
[ -d "$DOCS_DIR/n8n" ] && echo -e "  ${GREEN}[OK]${NC} n8n" || echo -e "  ${RED}[--]${NC} n8n"
echo ""
echo "CONFIGURATION:"
[ -d "$DOCS_DIR/config-formats" ] && echo -e "  ${GREEN}[OK]${NC} JSON Schema / YAML" || echo -e "  ${RED}[--]${NC} JSON Schema / YAML"
echo ""
echo "========================================================"

# Exit with error if any downloads failed
if [ "$FAILED_COUNT" -gt 0 ]; then
    exit 1
fi
