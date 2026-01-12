# Kubernetes Deployment

Production-ready Kubernetes manifests for the AI RAG Stack.

## Directory Structure

```
k8s/
├── namespace.yaml           # Dedicated namespace
├── configmap.yaml           # Non-sensitive configuration
├── secret.yaml              # Secret template (replace placeholders)
├── backend-deployment.yaml  # Backend API with probes
├── backend-service.yaml     # ClusterIP service
├── backend-hpa.yaml         # HorizontalPodAutoscaler (2-10 replicas)
├── backend-pdb.yaml         # PodDisruptionBudget
├── frontend-deployment.yaml # React frontend
├── qdrant-statefulset.yaml  # Vector database with persistence
├── redis-deployment.yaml    # Cache with persistence
├── postgres-statefulset.yaml# PostgreSQL database
├── ollama-deployment.yaml   # LLM inference (GPU)
├── ingress.yaml             # Ingress with TLS placeholder
├── networkpolicy.yaml       # Network isolation policies
├── kustomization.yaml       # Kustomize configuration
└── overlays/
    ├── dev/                 # Development overrides
    ├── staging/             # Staging overrides
    └── production/          # Production overrides
```

## Prerequisites

1. Kubernetes cluster (1.25+)
2. kubectl configured
3. NGINX Ingress Controller installed
4. (Optional) cert-manager for TLS
5. (Optional) GPU nodes with NVIDIA device plugin for Ollama

## Quick Start

### 1. Configure Secrets

Edit `secret.yaml` and replace placeholder values:

```bash
# Edit the secret file
vi k8s/secret.yaml

# Or use kubectl to create secrets directly
kubectl create namespace ai-rag-stack
kubectl -n ai-rag-stack create secret generic backend-secrets \
  --from-literal=POSTGRES_USER=raguser \
  --from-literal=POSTGRES_PASSWORD=<your-password> \
  --from-literal=TAVILY_API_KEY=<your-api-key>
```

### 2. Deploy with Kustomize

```bash
# Preview what will be deployed
kubectl kustomize k8s/

# Deploy base configuration
kubectl apply -k k8s/

# Or deploy specific environment
kubectl apply -k k8s/overlays/dev
kubectl apply -k k8s/overlays/staging
kubectl apply -k k8s/overlays/production
```

### 3. Verify Deployment

```bash
# Check all pods are running
kubectl -n ai-rag-stack get pods

# Check services
kubectl -n ai-rag-stack get svc

# Check ingress
kubectl -n ai-rag-stack get ingress

# View backend logs
kubectl -n ai-rag-stack logs -f deployment/backend
```

### 4. Pull LLM Model

```bash
# Exec into Ollama pod and pull model
kubectl -n ai-rag-stack exec -it deployment/ollama -- ollama pull llama3.1:8b
```

## Environment Overlays

### Development (`k8s/overlays/dev`)
- Single replica deployments
- Reduced resource limits
- Debug logging enabled
- Uses `dev` image tags

### Staging (`k8s/overlays/staging`)
- 2 replicas
- Moderate resources
- Info logging
- Uses `staging` image tags

### Production (`k8s/overlays/production`)
- 3+ replicas with HPA (up to 10)
- Full resource allocation
- TLS enabled
- Warning-level logging
- Authentication enabled
- Uses versioned image tags (e.g., `v1.0.0`)

## Configuration

### Ingress

Update the host in `ingress.yaml`:

```yaml
spec:
  rules:
    - host: your-domain.example.com  # Change this
```

For TLS, uncomment the TLS section and configure cert-manager:

```yaml
spec:
  tls:
    - hosts:
        - your-domain.example.com
      secretName: ai-rag-tls
```

### GPU Support (Ollama)

The Ollama deployment requires GPU nodes. Ensure:

1. NVIDIA device plugin is installed
2. Nodes have `nvidia.com/gpu.present=true` label
3. GPU toleration is configured

For CPU-only deployment, modify `ollama-deployment.yaml`:

```yaml
# Remove these lines:
resources:
  requests:
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"
nodeSelector:
  nvidia.com/gpu.present: "true"
tolerations:
  - key: nvidia.com/gpu
    ...
```

### Storage Classes

Update PVC storage classes to match your cluster:

```yaml
spec:
  storageClassName: your-storage-class
```

## Monitoring

The backend exposes Prometheus metrics at `/metrics`. Configure Prometheus to scrape:

```yaml
scrape_configs:
  - job_name: 'ai-rag-backend'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ai-rag-stack
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

## Security

### Network Policies

Network policies restrict traffic:
- Backend: accepts traffic from ingress and Prometheus only
- Qdrant/Redis/Postgres: accept traffic from backend only
- Egress: controlled per service

### Pod Security

- All pods run as non-root
- Read-only root filesystems where possible
- Capabilities dropped
- Security contexts enforced

### Secrets Management

For production, consider:
- [External Secrets Operator](https://external-secrets.io/)
- [Sealed Secrets](https://sealed-secrets.netlify.app/)
- [HashiCorp Vault](https://www.vaultproject.io/)

## Troubleshooting

### Pod not starting

```bash
# Check pod events
kubectl -n ai-rag-stack describe pod <pod-name>

# Check logs
kubectl -n ai-rag-stack logs <pod-name>
```

### Backend not connecting to services

```bash
# Test connectivity
kubectl -n ai-rag-stack exec -it deployment/backend -- nc -zv qdrant 6333
kubectl -n ai-rag-stack exec -it deployment/backend -- nc -zv redis 6379
```

### Ollama GPU issues

```bash
# Check GPU availability
kubectl -n ai-rag-stack exec -it deployment/ollama -- nvidia-smi
```

## Scaling

### Manual Scaling

```bash
kubectl -n ai-rag-stack scale deployment/backend --replicas=5
```

### HPA Configuration

The HPA scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)

Adjust in `backend-hpa.yaml` as needed.
