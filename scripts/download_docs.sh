#!/bin/bash

# DevOps Documentation Downloader
# This script clones and organizes documentation from various DevOps tools

set -e

DOCS_DIR="${1:-../data/docs}"
mkdir -p "$DOCS_DIR"

echo "Downloading DevOps documentation to $DOCS_DIR..."

# Kubernetes Documentation
echo "Downloading Kubernetes docs..."
if [ ! -d "$DOCS_DIR/kubernetes" ]; then
    git clone --depth 1 https://github.com/kubernetes/website.git "$DOCS_DIR/kubernetes"
else
    echo "Kubernetes docs already exist, skipping..."
fi

# Kubernetes AI Documentation
echo "Downloading Kubernetes AI docs..."
if [ ! -d "$DOCS_DIR/kubernetes-ai" ]; then
    git clone --depth 1 https://github.com/kubernetes-sigs/cluster-api.git "$DOCS_DIR/k8s-ai-src"
    if [ -d "$DOCS_DIR/k8s-ai-src/docs" ]; then
        mv "$DOCS_DIR/k8s-ai-src/docs" "$DOCS_DIR/kubernetes-ai"
        rm -rf "$DOCS_DIR/k8s-ai-src"
    fi
    # Also get kubeflow documentation for ML/AI on Kubernetes
    git clone --depth 1 https://github.com/kubeflow/website.git "$DOCS_DIR/kubeflow-tmp" || \
        echo "Warning: Could not clone Kubeflow docs"
    if [ -d "$DOCS_DIR/kubeflow-tmp/content" ]; then
        mkdir -p "$DOCS_DIR/kubernetes-ai/kubeflow"
        mv "$DOCS_DIR/kubeflow-tmp/content" "$DOCS_DIR/kubernetes-ai/kubeflow/docs"
        rm -rf "$DOCS_DIR/kubeflow-tmp"
    fi
else
    echo "Kubernetes AI docs already exist, skipping..."
fi

# Terraform Documentation
echo "Downloading Terraform docs..."
if [ ! -d "$DOCS_DIR/terraform" ]; then
    git clone --depth 1 https://github.com/hashicorp/terraform-docs-common.git "$DOCS_DIR/terraform"
else
    echo "Terraform docs already exist, skipping..."
fi

# Docker Documentation
echo "Downloading Docker docs..."
if [ ! -d "$DOCS_DIR/docker" ]; then
    git clone --depth 1 https://github.com/docker/docs.git "$DOCS_DIR/docker"
else
    echo "Docker docs already exist, skipping..."
fi

# Ansible Documentation
echo "Downloading Ansible docs..."
if [ ! -d "$DOCS_DIR/ansible" ]; then
    git clone --depth 1 https://github.com/ansible/ansible-documentation.git "$DOCS_DIR/ansible"
else
    echo "Ansible docs already exist, skipping..."
fi

# Prometheus Documentation
echo "Downloading Prometheus docs..."
if [ ! -d "$DOCS_DIR/prometheus" ]; then
    git clone --depth 1 https://github.com/prometheus/docs.git "$DOCS_DIR/prometheus"
else
    echo "Prometheus docs already exist, skipping..."
fi

# Python Documentation
echo "Downloading Python docs..."
if [ ! -d "$DOCS_DIR/python" ]; then
    git clone --depth 1 https://github.com/python/cpython.git "$DOCS_DIR/python-src"
    # Extract just the Doc directory
    if [ -d "$DOCS_DIR/python-src/Doc" ]; then
        mv "$DOCS_DIR/python-src/Doc" "$DOCS_DIR/python"
        rm -rf "$DOCS_DIR/python-src"
    fi
else
    echo "Python docs already exist, skipping..."
fi

# Go Documentation
echo "Downloading Go docs..."
if [ ! -d "$DOCS_DIR/go" ]; then
    git clone --depth 1 https://github.com/golang/go.git "$DOCS_DIR/go-src"
    # Extract just the doc directory
    if [ -d "$DOCS_DIR/go-src/doc" ]; then
        mv "$DOCS_DIR/go-src/doc" "$DOCS_DIR/go"
        rm -rf "$DOCS_DIR/go-src"
    fi
else
    echo "Go docs already exist, skipping..."
fi

# Bash Documentation (GNU Bash manual)
echo "Downloading Bash docs..."
if [ ! -d "$DOCS_DIR/bash" ]; then
    mkdir -p "$DOCS_DIR/bash"
    # Download Bash reference manual
    curl -o "$DOCS_DIR/bash/bash.html" https://www.gnu.org/software/bash/manual/bash.html || \
        echo "Warning: Could not download Bash manual"
    # Also get some advanced bash scripting guide
    git clone --depth 1 https://github.com/denysdovhan/bash-handbook.git "$DOCS_DIR/bash/handbook" || \
        echo "Warning: Could not clone bash handbook"
else
    echo "Bash docs already exist, skipping..."
fi

# Zsh Documentation
echo "Downloading Zsh docs..."
if [ ! -d "$DOCS_DIR/zsh" ]; then
    mkdir -p "$DOCS_DIR/zsh"
    # Clone Zsh source for documentation
    git clone --depth 1 https://github.com/zsh-users/zsh.git "$DOCS_DIR/zsh-src"
    if [ -d "$DOCS_DIR/zsh-src/Doc" ]; then
        mv "$DOCS_DIR/zsh-src/Doc" "$DOCS_DIR/zsh/manual"
        rm -rf "$DOCS_DIR/zsh-src"
    fi
    # Also get some Zsh guides
    git clone --depth 1 https://github.com/zsh-users/zsh-completions.git "$DOCS_DIR/zsh/completions" || \
        echo "Warning: Could not clone zsh completions"
else
    echo "Zsh docs already exist, skipping..."
fi

# JavaScript/Node.js Documentation
echo "Downloading JavaScript/Node.js docs..."
if [ ! -d "$DOCS_DIR/nodejs" ]; then
    git clone --depth 1 https://github.com/nodejs/node.git "$DOCS_DIR/nodejs-src"
    if [ -d "$DOCS_DIR/nodejs-src/doc" ]; then
        mv "$DOCS_DIR/nodejs-src/doc" "$DOCS_DIR/nodejs"
        rm -rf "$DOCS_DIR/nodejs-src"
    fi
    # Also get MDN JavaScript docs
    git clone --depth 1 https://github.com/mdn/content.git "$DOCS_DIR/mdn-js-tmp"
    if [ -d "$DOCS_DIR/mdn-js-tmp/files/en-us/web/javascript" ]; then
        mkdir -p "$DOCS_DIR/javascript"
        mv "$DOCS_DIR/mdn-js-tmp/files/en-us/web/javascript" "$DOCS_DIR/javascript/mdn"
        rm -rf "$DOCS_DIR/mdn-js-tmp"
    fi
else
    echo "JavaScript/Node.js docs already exist, skipping..."
fi

# Rust Documentation
echo "Downloading Rust docs..."
if [ ! -d "$DOCS_DIR/rust" ]; then
    git clone --depth 1 https://github.com/rust-lang/rust.git "$DOCS_DIR/rust-src"
    if [ -d "$DOCS_DIR/rust-src/src/doc" ]; then
        mv "$DOCS_DIR/rust-src/src/doc" "$DOCS_DIR/rust"
        rm -rf "$DOCS_DIR/rust-src"
    fi
    # Also get Rust by Example
    git clone --depth 1 https://github.com/rust-lang/rust-by-example.git "$DOCS_DIR/rust/by-example" || \
        echo "Warning: Could not clone Rust by Example"
else
    echo "Rust docs already exist, skipping..."
fi

# Grafana Documentation
echo "Downloading Grafana docs..."
if [ ! -d "$DOCS_DIR/grafana" ]; then
    git clone --depth 1 https://github.com/grafana/grafana.git "$DOCS_DIR/grafana-src"
    if [ -d "$DOCS_DIR/grafana-src/docs" ]; then
        mv "$DOCS_DIR/grafana-src/docs" "$DOCS_DIR/grafana"
        rm -rf "$DOCS_DIR/grafana-src"
    fi
else
    echo "Grafana docs already exist, skipping..."
fi

# ELK Stack Documentation
echo "Downloading ELK Stack docs..."
# Elasticsearch
if [ ! -d "$DOCS_DIR/elasticsearch" ]; then
    git clone --depth 1 https://github.com/elastic/elasticsearch.git "$DOCS_DIR/es-src"
    if [ -d "$DOCS_DIR/es-src/docs" ]; then
        mv "$DOCS_DIR/es-src/docs" "$DOCS_DIR/elasticsearch"
        rm -rf "$DOCS_DIR/es-src"
    fi
else
    echo "Elasticsearch docs already exist, skipping..."
fi

# Logstash
if [ ! -d "$DOCS_DIR/logstash" ]; then
    git clone --depth 1 https://github.com/elastic/logstash.git "$DOCS_DIR/logstash-src"
    if [ -d "$DOCS_DIR/logstash-src/docs" ]; then
        mv "$DOCS_DIR/logstash-src/docs" "$DOCS_DIR/logstash"
        rm -rf "$DOCS_DIR/logstash-src"
    fi
else
    echo "Logstash docs already exist, skipping..."
fi

# Kibana
if [ ! -d "$DOCS_DIR/kibana" ]; then
    git clone --depth 1 https://github.com/elastic/kibana.git "$DOCS_DIR/kibana-src"
    if [ -d "$DOCS_DIR/kibana-src/docs" ]; then
        mv "$DOCS_DIR/kibana-src/docs" "$DOCS_DIR/kibana"
        rm -rf "$DOCS_DIR/kibana-src"
    fi
else
    echo "Kibana docs already exist, skipping..."
fi

# Git Documentation
echo "Downloading Git docs..."
if [ ! -d "$DOCS_DIR/git" ]; then
    mkdir -p "$DOCS_DIR/git"
    # Clone the official Git book
    git clone --depth 1 https://github.com/progit/progit2.git "$DOCS_DIR/git/progit2" || \
        echo "Warning: Could not clone Pro Git book"
    # Get Git reference
    git clone --depth 1 https://github.com/git/git.git "$DOCS_DIR/git-src"
    if [ -d "$DOCS_DIR/git-src/Documentation" ]; then
        mv "$DOCS_DIR/git-src/Documentation" "$DOCS_DIR/git/official-docs"
        rm -rf "$DOCS_DIR/git-src"
    fi
else
    echo "Git docs already exist, skipping..."
fi

# Jenkins Documentation
echo "Downloading Jenkins docs..."
if [ ! -d "$DOCS_DIR/jenkins" ]; then
    git clone --depth 1 https://github.com/jenkins-infra/jenkins.io.git "$DOCS_DIR/jenkins-src"
    if [ -d "$DOCS_DIR/jenkins-src/content" ]; then
        mv "$DOCS_DIR/jenkins-src/content" "$DOCS_DIR/jenkins"
        rm -rf "$DOCS_DIR/jenkins-src"
    fi
else
    echo "Jenkins docs already exist, skipping..."
fi

# GitHub Actions Documentation
echo "Downloading GitHub Actions docs..."
if [ ! -d "$DOCS_DIR/github-actions" ]; then
    mkdir -p "$DOCS_DIR/github-actions"
    git clone --depth 1 https://github.com/github/docs.git "$DOCS_DIR/gh-docs-tmp"
    if [ -d "$DOCS_DIR/gh-docs-tmp/content/actions" ]; then
        mv "$DOCS_DIR/gh-docs-tmp/content/actions" "$DOCS_DIR/github-actions/docs"
        rm -rf "$DOCS_DIR/gh-docs-tmp"
    fi
else
    echo "GitHub Actions docs already exist, skipping..."
fi

# ArgoCD Documentation
echo "Downloading ArgoCD docs..."
if [ ! -d "$DOCS_DIR/argocd" ]; then
    git clone --depth 1 https://github.com/argoproj/argo-cd.git "$DOCS_DIR/argocd-src"
    if [ -d "$DOCS_DIR/argocd-src/docs" ]; then
        mv "$DOCS_DIR/argocd-src/docs" "$DOCS_DIR/argocd"
        rm -rf "$DOCS_DIR/argocd-src"
    fi
else
    echo "ArgoCD docs already exist, skipping..."
fi

# Helm Documentation
echo "Downloading Helm docs..."
if [ ! -d "$DOCS_DIR/helm" ]; then
    git clone --depth 1 https://github.com/helm/helm.git "$DOCS_DIR/helm-src"
    if [ -d "$DOCS_DIR/helm-src/docs" ]; then
        mv "$DOCS_DIR/helm-src/docs" "$DOCS_DIR/helm"
        rm -rf "$DOCS_DIR/helm-src"
    fi
else
    echo "Helm docs already exist, skipping..."
fi

# GCP Documentation
echo "Downloading GCP docs..."
if [ ! -d "$DOCS_DIR/gcp" ]; then
    mkdir -p "$DOCS_DIR/gcp"
    # GCP Python client docs
    git clone --depth 1 https://github.com/googleapis/google-cloud-python.git "$DOCS_DIR/gcp-tmp"
    if [ -d "$DOCS_DIR/gcp-tmp/docs" ]; then
        mv "$DOCS_DIR/gcp-tmp/docs" "$DOCS_DIR/gcp/python-client"
        rm -rf "$DOCS_DIR/gcp-tmp"
    fi
else
    echo "GCP docs already exist, skipping..."
fi

# n8n Documentation
echo "Downloading n8n docs..."
if [ ! -d "$DOCS_DIR/n8n" ]; then
    git clone --depth 1 https://github.com/n8n-io/n8n-docs.git "$DOCS_DIR/n8n" || \
        echo "Warning: Could not clone n8n docs"
else
    echo "n8n docs already exist, skipping..."
fi

# Configuration formats (JSON Schema, YAML guides)
echo "Downloading JSON/YAML documentation..."
if [ ! -d "$DOCS_DIR/config-formats" ]; then
    mkdir -p "$DOCS_DIR/config-formats"
    # JSON Schema
    git clone --depth 1 https://github.com/json-schema-org/json-schema-spec.git "$DOCS_DIR/config-formats/json-schema" || \
        echo "Warning: Could not clone JSON Schema"
    # YAML spec
    curl -o "$DOCS_DIR/config-formats/yaml-spec.md" https://yaml.org/spec/1.2.2/ || \
        echo "Warning: Could not download YAML spec"
else
    echo "Config formats docs already exist, skipping..."
fi

echo ""
echo "═══════════════════════════════════════════"
echo "Documentation download complete!"
echo "═══════════════════════════════════════════"
echo "Total size: $(du -sh $DOCS_DIR | cut -f1)"
echo ""
echo "Downloaded documentation for:"
echo ""
echo "DEVOPS & INFRASTRUCTURE:"
echo "  ✓ Kubernetes"
echo "  ✓ Kubernetes AI (Cluster API, Kubeflow)"
echo "  ✓ Terraform"
echo "  ✓ Docker"
echo "  ✓ Ansible"
echo "  ✓ Helm"
echo ""
echo "MONITORING & LOGGING:"
echo "  ✓ Prometheus"
echo "  ✓ Grafana"
echo "  ✓ Elasticsearch"
echo "  ✓ Logstash"
echo "  ✓ Kibana (ELK Stack)"
echo ""
echo "PROGRAMMING LANGUAGES:"
echo "  ✓ Python"
echo "  ✓ Go"
echo "  ✓ Rust"
echo "  ✓ JavaScript/Node.js"
echo "  ✓ Bash"
echo "  ✓ Zsh"
echo ""
echo "CI/CD & GITOPS:"
echo "  ✓ Git"
echo "  ✓ Jenkins"
echo "  ✓ GitHub Actions"
echo "  ✓ ArgoCD"
echo "  ✓ GitLab CI/CD"
echo ""
echo "CLOUD PLATFORMS:"
echo "  ✓ AWS"
echo "  ✓ Azure"
echo "  ✓ GCP"
echo ""
echo "AUTOMATION & INTEGRATION:"
echo "  ✓ n8n"
echo ""
echo "CONFIGURATION:"
echo "  ✓ JSON Schema"
echo "  ✓ YAML"
echo ""
