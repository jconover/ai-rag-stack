"""Conversation-aware query expansion module.

This module improves retrieval for follow-up questions by incorporating context
from recent conversation history. It resolves pronouns and references like
"it", "that", "the same thing" by extracting key terms from prior messages.

Usage:
    from app.conversation_context import conversation_expander

    # Get expanded query for retrieval
    expanded_query = conversation_expander.expand_query(
        query="How do I scale it?",
        conversation_history=[
            {"role": "user", "content": "What is a Kubernetes deployment?"},
            {"role": "assistant", "content": "A Kubernetes deployment manages..."},
        ]
    )
    # Result: "How do I scale Kubernetes deployment?"
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ConversationContextConfig:
    """Configuration for conversation-aware query expansion.

    Configuration is loaded from centralized settings in config.py,
    which reads from environment variables.
    """

    # Enable/disable conversation context for retrieval
    enabled: bool = None

    # Number of recent user messages to consider
    history_messages_limit: int = None

    # Minimum query length to trigger expansion (short queries need context)
    min_query_length_for_expansion: int = None

    # Maximum terms to extract from conversation history
    max_context_terms: int = None

    def __post_init__(self):
        """Load defaults from centralized settings if not explicitly set."""
        if self.enabled is None:
            self.enabled = getattr(settings, 'conversation_context_enabled', True)
        if self.history_messages_limit is None:
            self.history_messages_limit = getattr(settings, 'conversation_context_history_limit', 3)
        if self.min_query_length_for_expansion is None:
            self.min_query_length_for_expansion = getattr(settings, 'conversation_context_min_query_length', 5)
        if self.max_context_terms is None:
            self.max_context_terms = getattr(settings, 'conversation_context_max_terms', 10)


@dataclass
class ConversationContextResult:
    """Result container for conversation context expansion."""

    original_query: str
    expanded_query: str
    expanded: bool = False
    context_terms: List[str] = field(default_factory=list)
    resolved_references: Dict[str, str] = field(default_factory=dict)
    skip_reason: Optional[str] = None


class ConversationContextExpander:
    """Expands queries using conversation history for better retrieval.

    This class analyzes recent conversation history to:
    1. Extract key technical terms mentioned in prior exchanges
    2. Resolve pronouns and references (it, that, this, the same)
    3. Build context-aware search queries for improved retrieval

    Features:
    - DevOps-focused term extraction (Kubernetes, Docker, Terraform, etc.)
    - Pronoun and reference resolution
    - Configurable via environment variables
    - Detailed result tracking for debugging

    Example:
        expander = ConversationContextExpander()

        # Follow-up question with pronoun
        result = expander.expand_query(
            "How do I debug it?",
            [{"role": "user", "content": "My pods keep crashing"}]
        )
        # result.expanded_query: "How do I debug pods crashing Kubernetes?"
    """

    # Pronouns and references that indicate context dependency
    REFERENCE_PATTERNS = {
        # Direct pronouns
        'it': re.compile(r'\b(it|its)\b', re.IGNORECASE),
        'that': re.compile(r'\b(that|those)\b', re.IGNORECASE),
        'this': re.compile(r'\b(this|these)\b', re.IGNORECASE),
        'they': re.compile(r'\b(they|them|their)\b', re.IGNORECASE),

        # Reference phrases
        'the_same': re.compile(r'\b(the same|same thing|same one|the above)\b', re.IGNORECASE),
        'mentioned': re.compile(r'\b(mentioned|described|discussed|explained)\b', re.IGNORECASE),
        'previous': re.compile(r'\b(previous|earlier|before|last|prior)\b', re.IGNORECASE),
    }

    # DevOps technical terms to extract from conversation history
    # These are high-value terms that improve retrieval specificity
    DEVOPS_TERM_PATTERNS = [
        # Kubernetes resources
        re.compile(r'\b(pod|pods|deployment|deployments|service|services|'
                   r'configmap|configmaps|secret|secrets|namespace|namespaces|'
                   r'ingress|ingresses|statefulset|statefulsets|daemonset|daemonsets|'
                   r'replicaset|replicasets|cronjob|cronjobs|job|jobs|'
                   r'persistentvolume|persistentvolumeclaim|pv|pvc|'
                   r'node|nodes|cluster|clusters|container|containers|'
                   r'serviceaccount|networkpolicy|resourcequota|limitrange|'
                   r'horizontalpodautoscaler|hpa|verticalpodautoscaler|vpa)\b', re.IGNORECASE),

        # Container technologies
        re.compile(r'\b(docker|dockerfile|container|image|registry|'
                   r'podman|containerd|cri-o|buildah|kaniko|'
                   r'layer|volume|mount|compose|swarm)\b', re.IGNORECASE),

        # IaC and configuration
        re.compile(r'\b(terraform|ansible|helm|kustomize|'
                   r'chart|playbook|role|task|module|provider|'
                   r'state|plan|apply|destroy|workspace|'
                   r'variable|output|resource|data|locals)\b', re.IGNORECASE),

        # Cloud providers
        re.compile(r'\b(aws|amazon|ec2|s3|rds|lambda|iam|vpc|'
                   r'eks|ecs|fargate|cloudformation|'
                   r'gcp|google cloud|gke|compute engine|cloud run|'
                   r'azure|aks|arm|blob|'
                   r'digitalocean|linode|vultr)\b', re.IGNORECASE),

        # CI/CD
        re.compile(r'\b(jenkins|gitlab|github actions|circleci|travis|'
                   r'argocd|flux|spinnaker|tekton|'
                   r'pipeline|workflow|stage|step|job|'
                   r'build|deploy|release|rollout|rollback)\b', re.IGNORECASE),

        # Monitoring and observability
        re.compile(r'\b(prometheus|grafana|datadog|newrelic|'
                   r'elasticsearch|kibana|logstash|elk|'
                   r'jaeger|zipkin|opentelemetry|'
                   r'metric|metrics|log|logs|trace|traces|'
                   r'alert|alerting|dashboard|slo|sli|sla)\b', re.IGNORECASE),

        # Networking
        re.compile(r'\b(loadbalancer|load balancer|nginx|haproxy|envoy|'
                   r'istio|linkerd|service mesh|'
                   r'dns|ssl|tls|certificate|cert|'
                   r'firewall|security group|network policy|'
                   r'port|endpoint|url|domain)\b', re.IGNORECASE),

        # Common operations and concepts
        re.compile(r'\b(scale|scaling|autoscale|autoscaling|'
                   r'restart|crash|crashloop|oom|memory|cpu|'
                   r'permission|access|authentication|authorization|'
                   r'backup|restore|migrate|migration|'
                   r'debug|troubleshoot|error|issue|problem)\b', re.IGNORECASE),
    ]

    # Generic terms to filter out (too common to be useful context)
    STOP_TERMS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'must', 'need', 'want', 'like',
        'use', 'used', 'using', 'make', 'made', 'get', 'got', 'give', 'take',
        'go', 'went', 'come', 'came', 'see', 'look', 'find', 'know', 'think',
        'say', 'said', 'tell', 'ask', 'work', 'try', 'put', 'run', 'running',
        'what', 'how', 'why', 'when', 'where', 'which', 'who', 'whom',
        'yes', 'no', 'not', 'just', 'only', 'also', 'very', 'too', 'so',
        'and', 'but', 'or', 'if', 'then', 'else', 'for', 'with', 'without',
        'about', 'from', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'over', 'out', 'in', 'on', 'off',
        'up', 'down', 'here', 'there', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'than', 'any', 'many',
        'much', 'own', 'same', 'different', 'new', 'old', 'first', 'last',
        'long', 'great', 'little', 'good', 'bad', 'right', 'wrong',
        'thing', 'things', 'way', 'ways', 'time', 'times', 'part', 'parts',
        'file', 'files', 'command', 'commands', 'example', 'examples',
        'step', 'steps', 'following', 'below', 'above', 'please', 'thanks',
        'help', 'need', 'want', 'trying', 'getting', 'doing', 'making',
    }

    def __init__(self, config: Optional[ConversationContextConfig] = None):
        """Initialize the conversation context expander.

        Args:
            config: Optional ConversationContextConfig instance. If not provided,
                   configuration is loaded from environment variables.
        """
        self.config = config or ConversationContextConfig()

    @property
    def enabled(self) -> bool:
        """Check if conversation context is enabled."""
        return self.config.enabled

    def _has_reference(self, query: str) -> Tuple[bool, List[str]]:
        """Check if the query contains pronouns or references needing resolution.

        Args:
            query: The user's query

        Returns:
            Tuple of (has_reference, list of matched reference types)
        """
        matched_types = []
        for ref_type, pattern in self.REFERENCE_PATTERNS.items():
            if pattern.search(query):
                matched_types.append(ref_type)

        return bool(matched_types), matched_types

    def _extract_devops_terms(self, text: str) -> Set[str]:
        """Extract DevOps-related technical terms from text.

        Args:
            text: Text to extract terms from

        Returns:
            Set of extracted technical terms (lowercase)
        """
        terms = set()
        for pattern in self.DEVOPS_TERM_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                term = match.lower().strip()
                if term and term not in self.STOP_TERMS and len(term) > 2:
                    terms.add(term)
        return terms

    def _extract_context_from_history(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[Set[str], str]:
        """Extract key terms and primary topic from conversation history.

        Args:
            conversation_history: List of message dicts with 'role' and 'content'

        Returns:
            Tuple of (set of context terms, primary topic string)
        """
        all_terms = set()
        user_messages = []

        # Process messages in reverse chronological order (most recent first)
        # Limit to configured number of messages
        recent_messages = conversation_history[-self.config.history_messages_limit * 2:]

        for msg in recent_messages:
            role = msg.get('role', '')
            content = msg.get('content', '')

            if not content:
                continue

            # Extract terms from both user and assistant messages
            terms = self._extract_devops_terms(content)
            all_terms.update(terms)

            # Track user messages for topic identification
            if role == 'user':
                user_messages.append(content)

        # Identify primary topic from most recent user message
        primary_topic = ""
        if user_messages:
            # Get terms from most recent user message as primary topic
            recent_user_msg = user_messages[-1] if user_messages else ""
            recent_terms = self._extract_devops_terms(recent_user_msg)
            if recent_terms:
                # Prioritize Kubernetes and container terms
                k8s_terms = {'pod', 'pods', 'deployment', 'deployments', 'service',
                             'kubernetes', 'k8s', 'container', 'containers'}
                priority_terms = recent_terms.intersection(k8s_terms)
                if priority_terms:
                    primary_topic = " ".join(sorted(priority_terms)[:2])
                else:
                    primary_topic = " ".join(sorted(recent_terms)[:2])

        # Limit total terms
        limited_terms = set(sorted(all_terms)[:self.config.max_context_terms])

        return limited_terms, primary_topic

    def _resolve_references(
        self,
        query: str,
        primary_topic: str,
        context_terms: Set[str]
    ) -> Tuple[str, Dict[str, str]]:
        """Resolve pronouns and references in the query using context.

        Args:
            query: The user's query
            primary_topic: Primary topic from conversation history
            context_terms: Set of context terms from history

        Returns:
            Tuple of (resolved query, dict of resolutions made)
        """
        resolved_query = query
        resolutions = {}

        if not primary_topic:
            return resolved_query, resolutions

        # Replace common reference patterns with topic
        replacements = [
            (r'\b(it|its)\b', primary_topic),
            (r'\b(that)\b(?!\s+(?:is|was|are|were|has|have|can|could|will|would|should|must|may|might))',
             primary_topic),
            (r'\bthe same thing\b', primary_topic),
            (r'\bthe same\b', primary_topic),
            (r'\bthis\b(?!\s+(?:is|was|are|were|has|have|can|could|will|would|should|must|may|might))',
             primary_topic),
        ]

        for pattern, replacement in replacements:
            regex = re.compile(pattern, re.IGNORECASE)
            if regex.search(resolved_query):
                old_query = resolved_query
                resolved_query = regex.sub(replacement, resolved_query, count=1)
                if old_query != resolved_query:
                    resolutions[pattern] = replacement
                    break  # Only do one replacement to avoid over-substitution

        return resolved_query, resolutions

    def should_expand(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Determine if the query should be expanded with conversation context.

        Args:
            query: The user's query
            conversation_history: Optional list of prior messages

        Returns:
            Tuple of (should_expand, skip_reason if not expanding)
        """
        if not self.config.enabled:
            return False, "conversation_context_disabled"

        if not conversation_history:
            return False, "no_conversation_history"

        # Filter to only user messages for history check
        user_messages = [m for m in conversation_history if m.get('role') == 'user']
        if len(user_messages) < 1:
            return False, "no_prior_user_messages"

        query = query.strip()

        # Check if query contains references that need resolution
        has_ref, ref_types = self._has_reference(query)

        # Short queries with references definitely need expansion
        if has_ref and len(query) < 50:
            return True, None

        # Queries that are very short likely need context
        if len(query.split()) <= 5:
            return True, None

        # If query already has specific DevOps terms, might not need expansion
        query_terms = self._extract_devops_terms(query)
        if len(query_terms) >= 3:
            return False, "query_already_specific"

        # Default: expand if we have history and query is not too long
        if len(query) <= self.config.min_query_length_for_expansion * 20:
            return True, None

        return False, "query_too_long"

    def expand_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ConversationContextResult:
        """Expand query using conversation history for improved retrieval.

        Args:
            query: The user's current query
            conversation_history: List of prior messages with 'role' and 'content'

        Returns:
            ConversationContextResult with expanded query and metadata
        """
        result = ConversationContextResult(
            original_query=query,
            expanded_query=query
        )

        # Check if we should expand
        should_expand, skip_reason = self.should_expand(query, conversation_history)
        if not should_expand:
            result.skip_reason = skip_reason
            return result

        # Extract context from history
        context_terms, primary_topic = self._extract_context_from_history(
            conversation_history or []
        )

        if not context_terms and not primary_topic:
            result.skip_reason = "no_relevant_context_found"
            return result

        result.context_terms = list(context_terms)

        # Check for references needing resolution
        has_ref, ref_types = self._has_reference(query)

        if has_ref and primary_topic:
            # Resolve references using primary topic
            resolved_query, resolutions = self._resolve_references(
                query, primary_topic, context_terms
            )
            result.resolved_references = resolutions
            result.expanded_query = resolved_query
            result.expanded = True

            logger.debug(
                f"Resolved references in query: '{query}' -> '{resolved_query}' "
                f"(resolutions: {resolutions})"
            )
        else:
            # No references to resolve, but add context terms for better retrieval
            # Only add terms not already in the query
            query_lower = query.lower()
            new_terms = [t for t in context_terms if t.lower() not in query_lower]

            if new_terms:
                # Add most relevant terms (limit to avoid query bloat)
                terms_to_add = new_terms[:3]
                result.expanded_query = f"{query} {' '.join(terms_to_add)}"
                result.expanded = True

                logger.debug(
                    f"Added context terms to query: '{query}' -> '{result.expanded_query}'"
                )

        return result

    def build_context_aware_search_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build a context-aware search query for retrieval.

        Convenience method that returns just the expanded query string.

        Args:
            query: The user's current query
            conversation_history: List of prior messages

        Returns:
            Expanded query string ready for retrieval
        """
        result = self.expand_query(query, conversation_history)
        return result.expanded_query

    def get_status(self) -> Dict:
        """Get the current status of the conversation context expander.

        Returns:
            Dictionary with configuration and status information
        """
        return {
            'enabled': self.config.enabled,
            'history_messages_limit': self.config.history_messages_limit,
            'min_query_length_for_expansion': self.config.min_query_length_for_expansion,
            'max_context_terms': self.config.max_context_terms,
            'reference_patterns': list(self.REFERENCE_PATTERNS.keys()),
        }


# Singleton instance for use throughout the application
conversation_expander = ConversationContextExpander()


# Convenience functions for direct usage
def expand_with_context(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> ConversationContextResult:
    """Convenience function to expand a query using conversation context.

    Args:
        query: The user's current query
        conversation_history: List of prior messages

    Returns:
        ConversationContextResult with expanded query
    """
    return conversation_expander.expand_query(query, conversation_history)


def get_context_aware_query(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Convenience function to get a context-aware search query.

    Args:
        query: The user's current query
        conversation_history: List of prior messages

    Returns:
        Expanded query string
    """
    return conversation_expander.build_context_aware_search_query(
        query, conversation_history
    )
