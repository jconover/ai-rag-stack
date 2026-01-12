"""Prompt templates for common DevOps queries with variable support"""
import re
from typing import Any, Dict, List, Optional, Tuple


PROMPT_TEMPLATES = [
    {
        "id": "k8s-debug-pod",
        "category": "Kubernetes",
        "title": "Debug Pod Issues",
        "description": "Troubleshoot Kubernetes pod problems",
        "prompt": "My Kubernetes pod {pod_name} in namespace {namespace} is not starting. How do I debug it? Please include commands to check logs, events, and pod status.",
        "variables": [
            {
                "name": "pod_name",
                "type": "string",
                "description": "Name of the pod to debug",
                "required": False,
                "default": "<pod-name>",
                "placeholder": "my-app-pod-xyz"
            },
            {
                "name": "namespace",
                "type": "string",
                "description": "Kubernetes namespace where the pod is running",
                "required": False,
                "default": "default",
                "placeholder": "default"
            }
        ]
    },
    {
        "id": "k8s-deployment",
        "category": "Kubernetes",
        "title": "Create Deployment",
        "description": "Generate a Kubernetes deployment",
        "prompt": "Create a Kubernetes deployment YAML for {app_name} with {replicas} replicas, resource limits (CPU: {cpu_limit}, Memory: {memory_limit}), and a liveness probe on port {probe_port}.",
        "variables": [
            {
                "name": "app_name",
                "type": "string",
                "description": "Name of the application",
                "required": True,
                "placeholder": "my-web-app"
            },
            {
                "name": "replicas",
                "type": "number",
                "description": "Number of pod replicas",
                "required": False,
                "default": 3
            },
            {
                "name": "cpu_limit",
                "type": "string",
                "description": "CPU resource limit",
                "required": False,
                "default": "500m",
                "placeholder": "500m"
            },
            {
                "name": "memory_limit",
                "type": "string",
                "description": "Memory resource limit",
                "required": False,
                "default": "512Mi",
                "placeholder": "512Mi"
            },
            {
                "name": "probe_port",
                "type": "number",
                "description": "Port for liveness probe",
                "required": False,
                "default": 8080
            }
        ]
    },
    {
        "id": "k8s-ai-ml",
        "category": "Kubernetes",
        "title": "AI/ML on Kubernetes",
        "description": "Deploy ML workloads with Kubeflow",
        "prompt": "How do I deploy {ml_framework} machine learning workloads on Kubernetes using Kubeflow? Include information about {workflow_type}.",
        "variables": [
            {
                "name": "ml_framework",
                "type": "select",
                "description": "ML framework to deploy",
                "required": False,
                "default": "TensorFlow",
                "options": ["TensorFlow", "PyTorch", "XGBoost", "Scikit-learn", "custom"]
            },
            {
                "name": "workflow_type",
                "type": "select",
                "description": "Type of ML workflow",
                "required": False,
                "default": "training, serving, and pipelines",
                "options": ["training", "serving", "pipelines", "training, serving, and pipelines"]
            }
        ]
    },
    {
        "id": "terraform-vpc",
        "category": "Terraform",
        "title": "AWS VPC Setup",
        "description": "Create VPC infrastructure",
        "prompt": "Write Terraform code to create an AWS VPC in region {aws_region} with {public_subnet_count} public and {private_subnet_count} private subnets, {include_nat} NAT gateway, and internet gateway. Use CIDR block {vpc_cidr}.",
        "variables": [
            {
                "name": "aws_region",
                "type": "string",
                "description": "AWS region for the VPC",
                "required": False,
                "default": "us-east-1",
                "placeholder": "us-east-1"
            },
            {
                "name": "vpc_cidr",
                "type": "string",
                "description": "CIDR block for the VPC",
                "required": False,
                "default": "10.0.0.0/16",
                "placeholder": "10.0.0.0/16"
            },
            {
                "name": "public_subnet_count",
                "type": "number",
                "description": "Number of public subnets",
                "required": False,
                "default": 2
            },
            {
                "name": "private_subnet_count",
                "type": "number",
                "description": "Number of private subnets",
                "required": False,
                "default": 2
            },
            {
                "name": "include_nat",
                "type": "select",
                "description": "Whether to include NAT gateway",
                "required": False,
                "default": "a",
                "options": ["a", "no"]
            }
        ]
    },
    {
        "id": "terraform-debug",
        "category": "Terraform",
        "title": "Debug Terraform",
        "description": "Troubleshoot Terraform issues",
        "prompt": "My Terraform apply is failing with error: {error_message}. I'm using Terraform version {tf_version} with {provider} provider. What are the debugging steps and how do I fix this?",
        "variables": [
            {
                "name": "error_message",
                "type": "string",
                "description": "The error message from Terraform",
                "required": False,
                "default": "[paste your error here]",
                "placeholder": "Error: Failed to create resource..."
            },
            {
                "name": "tf_version",
                "type": "string",
                "description": "Terraform version",
                "required": False,
                "default": "1.5+",
                "placeholder": "1.5.0"
            },
            {
                "name": "provider",
                "type": "select",
                "description": "Cloud provider",
                "required": False,
                "default": "AWS",
                "options": ["AWS", "GCP", "Azure", "Kubernetes", "other"]
            }
        ]
    },
    {
        "id": "docker-optimize",
        "category": "Docker",
        "title": "Optimize Dockerfile",
        "description": "Improve Docker image size and build time",
        "prompt": "How can I optimize my Dockerfile for a {language} application to reduce image size (current size: {current_size}) and improve build time? Please provide best practices including multi-stage builds and layer caching.",
        "variables": [
            {
                "name": "language",
                "type": "select",
                "description": "Programming language of the application",
                "required": False,
                "default": "Python",
                "options": ["Python", "Node.js", "Go", "Java", "Rust", ".NET", "Ruby"]
            },
            {
                "name": "current_size",
                "type": "string",
                "description": "Current Docker image size",
                "required": False,
                "default": "unknown",
                "placeholder": "1.2GB"
            }
        ]
    },
    {
        "id": "docker-compose",
        "category": "Docker",
        "title": "Docker Compose Setup",
        "description": "Create docker-compose configuration",
        "prompt": "Create a docker-compose.yml for a {app_type} application with {database} database, {cache} cache, and {reverse_proxy} reverse proxy. Include health checks and proper networking.",
        "variables": [
            {
                "name": "app_type",
                "type": "string",
                "description": "Type of application (e.g., web app, API)",
                "required": False,
                "default": "web app",
                "placeholder": "REST API"
            },
            {
                "name": "database",
                "type": "select",
                "description": "Database to use",
                "required": False,
                "default": "PostgreSQL",
                "options": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "none"]
            },
            {
                "name": "cache",
                "type": "select",
                "description": "Caching layer",
                "required": False,
                "default": "Redis",
                "options": ["Redis", "Memcached", "none"]
            },
            {
                "name": "reverse_proxy",
                "type": "select",
                "description": "Reverse proxy/load balancer",
                "required": False,
                "default": "Nginx",
                "options": ["Nginx", "Traefik", "HAProxy", "Caddy", "none"]
            }
        ]
    },
    {
        "id": "ansible-playbook",
        "category": "Ansible",
        "title": "Create Playbook",
        "description": "Generate an Ansible playbook",
        "prompt": "Create an Ansible playbook to install and configure {service} on {target_os} servers{ssl_config}. Include proper error handling and idempotency.",
        "variables": [
            {
                "name": "service",
                "type": "string",
                "description": "Service to install and configure",
                "required": False,
                "default": "Nginx",
                "placeholder": "Nginx"
            },
            {
                "name": "target_os",
                "type": "select",
                "description": "Target operating system",
                "required": False,
                "default": "Ubuntu",
                "options": ["Ubuntu", "CentOS", "RHEL", "Debian", "Amazon Linux"]
            },
            {
                "name": "ssl_config",
                "type": "select",
                "description": "SSL certificate configuration",
                "required": False,
                "default": " with Let's Encrypt SSL certificates",
                "options": [" with Let's Encrypt SSL certificates", " with self-signed certificates", ""]
            }
        ]
    },
    {
        "id": "prometheus-query",
        "category": "Monitoring",
        "title": "Prometheus Query",
        "description": "Write PromQL queries",
        "prompt": "Write Prometheus queries to monitor {metric_types} for {service_type}. Include alert thresholds and Grafana panel recommendations.",
        "variables": [
            {
                "name": "metric_types",
                "type": "string",
                "description": "Types of metrics to monitor",
                "required": False,
                "default": "CPU usage, memory usage, and request rate",
                "placeholder": "error rates and latency"
            },
            {
                "name": "service_type",
                "type": "select",
                "description": "Type of service to monitor",
                "required": False,
                "default": "a web service",
                "options": ["a web service", "a database", "a message queue", "a Kubernetes cluster", "a microservices application"]
            }
        ]
    },
    {
        "id": "grafana-dashboard",
        "category": "Monitoring",
        "title": "Grafana Dashboard",
        "description": "Design monitoring dashboard",
        "prompt": "What metrics and panels should I include in a Grafana dashboard for monitoring {target_system}? I want to track {focus_areas}.",
        "variables": [
            {
                "name": "target_system",
                "type": "select",
                "description": "System to monitor",
                "required": False,
                "default": "a Kubernetes cluster",
                "options": ["a Kubernetes cluster", "a PostgreSQL database", "an Nginx server", "a Redis cache", "a microservices application"]
            },
            {
                "name": "focus_areas",
                "type": "string",
                "description": "Key areas to focus on",
                "required": False,
                "default": "performance, availability, and resource usage",
                "placeholder": "latency and error rates"
            }
        ]
    },
    {
        "id": "ci-pipeline",
        "category": "CI/CD",
        "title": "CI Pipeline",
        "description": "Create CI/CD pipeline",
        "prompt": "Create a {ci_platform} workflow that builds a {language} application as a Docker image, runs {test_types} tests, and deploys to {deploy_target}.",
        "variables": [
            {
                "name": "ci_platform",
                "type": "select",
                "description": "CI/CD platform",
                "required": False,
                "default": "GitHub Actions",
                "options": ["GitHub Actions", "GitLab CI", "Jenkins", "CircleCI", "Azure DevOps"]
            },
            {
                "name": "language",
                "type": "select",
                "description": "Programming language",
                "required": False,
                "default": "Python",
                "options": ["Python", "Node.js", "Go", "Java", "Rust", ".NET"]
            },
            {
                "name": "test_types",
                "type": "string",
                "description": "Types of tests to run",
                "required": False,
                "default": "unit and integration",
                "placeholder": "unit, integration, and e2e"
            },
            {
                "name": "deploy_target",
                "type": "select",
                "description": "Deployment target",
                "required": False,
                "default": "Kubernetes",
                "options": ["Kubernetes", "AWS ECS", "AWS Lambda", "Google Cloud Run", "Azure Container Apps"]
            }
        ]
    },
    {
        "id": "explain-error",
        "category": "Debugging",
        "title": "Explain Error",
        "description": "Understand and fix errors",
        "prompt": "I'm getting this error in my {context}: {error_message}. What does it mean and how do I fix it?",
        "variables": [
            {
                "name": "context",
                "type": "string",
                "description": "Where the error occurred (e.g., application, deployment)",
                "required": False,
                "default": "application",
                "placeholder": "Kubernetes deployment"
            },
            {
                "name": "error_message",
                "type": "string",
                "description": "The error message",
                "required": True,
                "placeholder": "[paste your error here]"
            }
        ]
    },
    {
        "id": "security-best-practices",
        "category": "Security",
        "title": "Security Best Practices",
        "description": "Get security recommendations",
        "prompt": "What are the security best practices for deploying {application_type} applications in {platform}? Focus on {security_areas}.",
        "variables": [
            {
                "name": "application_type",
                "type": "string",
                "description": "Type of application",
                "required": False,
                "default": "containerized",
                "placeholder": "microservices"
            },
            {
                "name": "platform",
                "type": "select",
                "description": "Deployment platform",
                "required": False,
                "default": "Kubernetes",
                "options": ["Kubernetes", "AWS", "GCP", "Azure", "on-premises"]
            },
            {
                "name": "security_areas",
                "type": "string",
                "description": "Security areas to focus on",
                "required": False,
                "default": "RBAC, network policies, and secrets management",
                "placeholder": "authentication and encryption"
            }
        ]
    },
    {
        "id": "python-script",
        "category": "Scripting",
        "title": "Python Automation",
        "description": "Create automation script",
        "prompt": "Write a Python script to {task_description}. Include error handling, logging, and {additional_requirements}.",
        "variables": [
            {
                "name": "task_description",
                "type": "string",
                "description": "What the script should do",
                "required": True,
                "placeholder": "automate backup of PostgreSQL databases to S3"
            },
            {
                "name": "additional_requirements",
                "type": "string",
                "description": "Additional requirements for the script",
                "required": False,
                "default": "proper documentation",
                "placeholder": "retry logic and notifications"
            }
        ]
    },
    {
        "id": "bash-script",
        "category": "Scripting",
        "title": "Bash Script",
        "description": "Create shell script",
        "prompt": "Write a Bash script to {task_description}. Include error handling, make it idempotent, and {additional_features}.",
        "variables": [
            {
                "name": "task_description",
                "type": "string",
                "description": "What the script should do",
                "required": True,
                "placeholder": "set up a development environment"
            },
            {
                "name": "additional_features",
                "type": "string",
                "description": "Additional features to include",
                "required": False,
                "default": "add logging and validation",
                "placeholder": "support for dry-run mode"
            }
        ]
    },
    {
        "id": "backup-strategy",
        "category": "Operations",
        "title": "Backup Strategy",
        "description": "Design backup solution",
        "prompt": "Design a backup and disaster recovery strategy for a production {application_type} application running on {platform} with {database_type}. Consider RPO of {rpo} and RTO of {rto}.",
        "variables": [
            {
                "name": "application_type",
                "type": "string",
                "description": "Type of application",
                "required": False,
                "default": "web",
                "placeholder": "e-commerce"
            },
            {
                "name": "platform",
                "type": "select",
                "description": "Deployment platform",
                "required": False,
                "default": "Kubernetes",
                "options": ["Kubernetes", "AWS EC2", "AWS ECS", "GCP GKE", "Azure AKS"]
            },
            {
                "name": "database_type",
                "type": "select",
                "description": "Database system",
                "required": False,
                "default": "PostgreSQL",
                "options": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "multiple databases"]
            },
            {
                "name": "rpo",
                "type": "select",
                "description": "Recovery Point Objective (max data loss)",
                "required": False,
                "default": "1 hour",
                "options": ["5 minutes", "15 minutes", "1 hour", "4 hours", "24 hours"]
            },
            {
                "name": "rto",
                "type": "select",
                "description": "Recovery Time Objective (max downtime)",
                "required": False,
                "default": "1 hour",
                "options": ["15 minutes", "1 hour", "4 hours", "24 hours"]
            }
        ]
    }
]


def get_templates() -> List[Dict[str, Any]]:
    """Get all prompt templates"""
    return PROMPT_TEMPLATES


def get_template_by_id(template_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific template by ID"""
    for template in PROMPT_TEMPLATES:
        if template['id'] == template_id:
            return template
    return None


def get_templates_by_category(category: str) -> List[Dict[str, Any]]:
    """Get templates filtered by category"""
    return [t for t in PROMPT_TEMPLATES if t['category'] == category]


def get_categories() -> List[str]:
    """Get all unique categories"""
    return list(set(t['category'] for t in PROMPT_TEMPLATES))


def _extract_variables_from_prompt(prompt: str) -> List[str]:
    """Extract variable names from a prompt template.

    Returns list of variable names found in {variable} format.
    """
    # Match {variable_name} patterns, excluding escaped braces {{}}
    pattern = r'(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})'
    return re.findall(pattern, prompt)


def _get_variable_definition(template: Dict[str, Any], var_name: str) -> Optional[Dict[str, Any]]:
    """Get the variable definition from template by variable name."""
    variables = template.get('variables', [])
    for var in variables:
        if var.get('name') == var_name:
            return var
    return None


def validate_template_variables(
    template: Dict[str, Any],
    provided_variables: Dict[str, Any]
) -> Tuple[bool, List[str], List[str]]:
    """Validate provided variables against template requirements.

    Args:
        template: Template dictionary with variables definition
        provided_variables: Dictionary of variable name -> value

    Returns:
        Tuple of (is_valid, missing_required, unknown_variables)
    """
    template_vars = template.get('variables', [])
    variable_names = {v['name'] for v in template_vars}
    required_vars = {v['name'] for v in template_vars if v.get('required', False)}

    # Check for missing required variables
    provided_names = set(provided_variables.keys())
    missing_required = list(required_vars - provided_names)

    # Check for unknown variables
    unknown_variables = list(provided_names - variable_names)

    is_valid = len(missing_required) == 0

    return is_valid, missing_required, unknown_variables


def render_template(
    template: Dict[str, Any],
    variables: Optional[Dict[str, Any]] = None,
    strict: bool = False
) -> Tuple[str, Dict[str, Any], List[str]]:
    """Render a template by substituting variables.

    Args:
        template: Template dictionary with prompt and variables definition
        variables: Dictionary of variable name -> value for substitution
        strict: If True, raise error for missing required variables

    Returns:
        Tuple of (rendered_prompt, variables_used, missing_required)

    Raises:
        ValueError: If strict=True and required variables are missing
    """
    if variables is None:
        variables = {}

    prompt = template.get('prompt', '')
    template_vars = template.get('variables', [])

    # Build a mapping of variable name -> value to use
    variables_used = {}
    missing_required = []

    for var_def in template_vars:
        var_name = var_def['name']
        is_required = var_def.get('required', False)
        default_value = var_def.get('default')

        if var_name in variables:
            # User provided value
            value = variables[var_name]
            # Type coercion for safety
            var_type = var_def.get('type', 'string')
            if var_type == 'number' and not isinstance(value, (int, float)):
                try:
                    value = float(value) if '.' in str(value) else int(value)
                except (ValueError, TypeError):
                    pass  # Keep original value
            elif var_type == 'boolean' and not isinstance(value, bool):
                if isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes')
            variables_used[var_name] = value
        elif default_value is not None:
            # Use default value
            variables_used[var_name] = default_value
        else:
            # No value provided and no default
            if is_required:
                missing_required.append(var_name)
                if strict:
                    raise ValueError(f"Required variable '{var_name}' not provided")
            # Use placeholder format for missing optional variables
            variables_used[var_name] = f"<{var_name}>"

    # Also handle any variables in the prompt that aren't defined
    prompt_vars = _extract_variables_from_prompt(prompt)
    for var_name in prompt_vars:
        if var_name not in variables_used:
            if var_name in variables:
                variables_used[var_name] = variables[var_name]
            else:
                variables_used[var_name] = f"<{var_name}>"

    # Render the prompt
    rendered_prompt = prompt
    for var_name, value in variables_used.items():
        rendered_prompt = rendered_prompt.replace(f"{{{var_name}}}", str(value))

    return rendered_prompt, variables_used, missing_required


def render_template_by_id(
    template_id: str,
    variables: Optional[Dict[str, Any]] = None,
    strict: bool = False
) -> Optional[Tuple[str, Dict[str, Any], List[str]]]:
    """Render a template by its ID.

    Args:
        template_id: Template identifier
        variables: Dictionary of variable name -> value
        strict: If True, raise error for missing required variables

    Returns:
        Tuple of (rendered_prompt, variables_used, missing_required) or None if not found
    """
    template = get_template_by_id(template_id)
    if template is None:
        return None
    return render_template(template, variables, strict)
