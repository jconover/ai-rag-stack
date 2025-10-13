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

echo "Documentation download complete!"
echo "Total size: $(du -sh $DOCS_DIR | cut -f1)"
