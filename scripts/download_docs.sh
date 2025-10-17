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

echo ""
echo "Documentation download complete!"
echo "Total size: $(du -sh $DOCS_DIR | cut -f1)"
echo ""
echo "Downloaded documentation for:"
echo "  - Kubernetes"
echo "  - Terraform"
echo "  - Docker"
echo "  - Ansible"
echo "  - Prometheus"
echo "  - Python"
echo "  - Go"
echo "  - Bash"
echo "  - Zsh"
