"""Prompt templates for common DevOps queries"""

PROMPT_TEMPLATES = [
    {
        "id": "k8s-debug-pod",
        "category": "Kubernetes",
        "title": "Debug Pod Issues",
        "description": "Troubleshoot Kubernetes pod problems",
        "prompt": "My Kubernetes pod is not starting. How do I debug it? Please include commands to check logs, events, and pod status."
    },
    {
        "id": "k8s-deployment",
        "category": "Kubernetes",
        "title": "Create Deployment",
        "description": "Generate a Kubernetes deployment",
        "prompt": "Create a Kubernetes deployment YAML for a web application with 3 replicas, resource limits, and a liveness probe."
    },
    {
        "id": "terraform-vpc",
        "category": "Terraform",
        "title": "AWS VPC Setup",
        "description": "Create VPC infrastructure",
        "prompt": "Write Terraform code to create an AWS VPC with public and private subnets, NAT gateway, and internet gateway."
    },
    {
        "id": "terraform-debug",
        "category": "Terraform",
        "title": "Debug Terraform",
        "description": "Troubleshoot Terraform issues",
        "prompt": "My Terraform apply is failing. What are the common debugging steps and commands I should use?"
    },
    {
        "id": "docker-optimize",
        "category": "Docker",
        "title": "Optimize Dockerfile",
        "description": "Improve Docker image size and build time",
        "prompt": "How can I optimize my Dockerfile to reduce image size and improve build time? Please provide best practices."
    },
    {
        "id": "docker-compose",
        "category": "Docker",
        "title": "Docker Compose Setup",
        "description": "Create docker-compose configuration",
        "prompt": "Create a docker-compose.yml for a web app with a PostgreSQL database, Redis cache, and Nginx reverse proxy."
    },
    {
        "id": "ansible-playbook",
        "category": "Ansible",
        "title": "Create Playbook",
        "description": "Generate an Ansible playbook",
        "prompt": "Create an Ansible playbook to install and configure Nginx on Ubuntu servers with SSL certificates."
    },
    {
        "id": "prometheus-query",
        "category": "Monitoring",
        "title": "Prometheus Query",
        "description": "Write PromQL queries",
        "prompt": "Write Prometheus queries to monitor CPU usage, memory usage, and request rate for a web service."
    },
    {
        "id": "grafana-dashboard",
        "category": "Monitoring",
        "title": "Grafana Dashboard",
        "description": "Design monitoring dashboard",
        "prompt": "What metrics and panels should I include in a Grafana dashboard for monitoring a Kubernetes cluster?"
    },
    {
        "id": "ci-pipeline",
        "category": "CI/CD",
        "title": "CI Pipeline",
        "description": "Create CI/CD pipeline",
        "prompt": "Create a GitHub Actions workflow that builds a Docker image, runs tests, and deploys to Kubernetes."
    },
    {
        "id": "explain-error",
        "category": "Debugging",
        "title": "Explain Error",
        "description": "Understand and fix errors",
        "prompt": "I'm getting this error: [paste your error here]. What does it mean and how do I fix it?"
    },
    {
        "id": "security-best-practices",
        "category": "Security",
        "title": "Security Best Practices",
        "description": "Get security recommendations",
        "prompt": "What are the security best practices for deploying applications in Kubernetes? Include RBAC, network policies, and secrets management."
    },
    {
        "id": "python-script",
        "category": "Scripting",
        "title": "Python Automation",
        "description": "Create automation script",
        "prompt": "Write a Python script to automate [describe your task]. Include error handling and logging."
    },
    {
        "id": "bash-script",
        "category": "Scripting",
        "title": "Bash Script",
        "description": "Create shell script",
        "prompt": "Write a Bash script to automate [describe your task]. Include error handling and make it idempotent."
    },
    {
        "id": "backup-strategy",
        "category": "Operations",
        "title": "Backup Strategy",
        "description": "Design backup solution",
        "prompt": "Design a backup and disaster recovery strategy for a production application running on Kubernetes with PostgreSQL."
    }
]


def get_templates():
    """Get all prompt templates"""
    return PROMPT_TEMPLATES


def get_template_by_id(template_id: str):
    """Get a specific template by ID"""
    for template in PROMPT_TEMPLATES:
        if template['id'] == template_id:
            return template
    return None


def get_templates_by_category(category: str):
    """Get templates filtered by category"""
    return [t for t in PROMPT_TEMPLATES if t['category'] == category]


def get_categories():
    """Get all unique categories"""
    return list(set(t['category'] for t in PROMPT_TEMPLATES))
