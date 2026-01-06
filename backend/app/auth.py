"""Authentication service for the DevOps AI Assistant.

This module provides comprehensive authentication functionality including:
- Password hashing and verification using bcrypt
- Secure token generation for sessions and API keys
- User registration and authentication
- Session management (create, validate, invalidate)
- API key management (create, validate, revoke)
- FastAPI dependencies for protected routes

Security considerations:
- Passwords are hashed with bcrypt (cost factor 12)
- Session tokens and API keys are generated with secrets.token_urlsafe
- Tokens are stored as SHA-256 hashes (never plaintext)
- Session expiration is enforced on validation
- API keys support optional expiration and permissions

Usage:
    from app.auth import auth_service, get_current_user, get_optional_user

    # In FastAPI endpoints
    @app.get("/protected")
    async def protected_route(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
    ):
        return {"user": user.username}

    # Optional authentication
    @app.get("/public")
    async def public_route(
        user: Optional[User] = Depends(get_optional_user),
        db: AsyncSession = Depends(get_db)
    ):
        if user:
            return {"user": user.username}
        return {"user": "anonymous"}
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import bcrypt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.db_models import User, UserSession, APIKey

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

SESSION_TOKEN_BYTES = 32  # 256 bits of entropy
API_KEY_BYTES = 32  # 256 bits of entropy
API_KEY_PREFIX = "rag_"  # Prefix for API keys
BCRYPT_ROUNDS = 12  # Cost factor for bcrypt
DEFAULT_SESSION_DURATION_HOURS = 24  # Default session lifetime


# =============================================================================
# Password Handling
# =============================================================================


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Uses bcrypt with a configurable cost factor (default 12 rounds).
    The salt is automatically generated and included in the hash.

    Args:
        password: Plain text password to hash

    Returns:
        Bcrypt hash string (includes algorithm, cost, salt, and hash)

    Example:
        >>> hashed = hash_password("my_secure_password")
        >>> hashed.startswith("$2b$")
        True
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its bcrypt hash.

    Performs constant-time comparison to prevent timing attacks.

    Args:
        password: Plain text password to verify
        password_hash: Bcrypt hash to compare against

    Returns:
        True if password matches, False otherwise

    Example:
        >>> hashed = hash_password("my_password")
        >>> verify_password("my_password", hashed)
        True
        >>> verify_password("wrong_password", hashed)
        False
    """
    try:
        password_bytes = password.encode("utf-8")
        hash_bytes = password_hash.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception as e:
        logger.warning(f"Password verification failed: {e}")
        return False


# =============================================================================
# Token Generation
# =============================================================================


def generate_session_token() -> str:
    """Generate a cryptographically secure session token.

    Uses secrets.token_urlsafe for secure random generation.
    Tokens are URL-safe base64 encoded.

    Returns:
        Secure random token string (43 characters for 32 bytes)

    Example:
        >>> token = generate_session_token()
        >>> len(token) >= 32
        True
    """
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def generate_api_key() -> str:
    """Generate a cryptographically secure API key with prefix.

    The prefix helps identify the key type and source.
    Format: rag_<random_token>

    Returns:
        API key string with prefix (e.g., "rag_abc123...")

    Example:
        >>> key = generate_api_key()
        >>> key.startswith("rag_")
        True
    """
    random_part = secrets.token_urlsafe(API_KEY_BYTES)
    return f"{API_KEY_PREFIX}{random_part}"


def hash_token(token: str) -> str:
    """Hash a token using SHA-256 for storage.

    Tokens should never be stored in plaintext. This function
    creates a deterministic hash for lookup purposes.

    Args:
        token: The token to hash

    Returns:
        Hexadecimal SHA-256 hash (64 characters)

    Example:
        >>> hash1 = hash_token("my_token")
        >>> hash2 = hash_token("my_token")
        >>> hash1 == hash2
        True
        >>> len(hash1)
        64
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# =============================================================================
# Authentication Service Class
# =============================================================================


class AuthService:
    """Service class for authentication operations.

    Provides methods for user registration, authentication, session
    management, and API key management. All methods are async and
    require a database session.

    Usage:
        auth = AuthService()

        # Register a user
        user = await auth.register_user(db, "user@example.com", "username", "password")

        # Authenticate
        user = await auth.authenticate(db, "user@example.com", "password")

        # Create session
        token = await auth.create_session(db, user.id, "127.0.0.1", "Mozilla/5.0")

        # Validate session
        user = await auth.validate_session(db, token)

        # Logout
        await auth.invalidate_session(db, token)
    """

    async def register_user(
        self,
        db: AsyncSession,
        email: str,
        username: str,
        password: str,
    ) -> User:
        """Register a new user account.

        Creates a new user with hashed password. Email and username
        must be unique.

        Args:
            db: Database session
            email: User's email address
            username: Unique username
            password: Plain text password (will be hashed)

        Returns:
            Created User instance

        Raises:
            HTTPException: If email or username already exists
        """
        # Normalize email to lowercase
        email_normalized = email.lower().strip()

        # Check for existing email
        result = await db.execute(
            select(User).where(User.email == email_normalized)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Check for existing username
        result = await db.execute(
            select(User).where(User.username == username)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken",
            )

        # Create user with hashed password
        user = User(
            email=email_normalized,
            username=username,
            password_hash=hash_password(password),
            is_active=True,
            is_verified=False,
        )
        db.add(user)
        await db.flush()  # Get the ID without committing
        await db.refresh(user)

        logger.info(f"Registered new user: {username} ({email_normalized})")
        return user

    async def authenticate(
        self,
        db: AsyncSession,
        email_or_username: str,
        password: str,
    ) -> User:
        """Authenticate a user by email/username and password.

        Supports login with either email or username. Updates
        last_login_at timestamp on successful authentication.

        Args:
            db: Database session
            email_or_username: Email address or username
            password: Plain text password

        Returns:
            Authenticated User instance

        Raises:
            HTTPException: If credentials are invalid or account is inactive
        """
        # Normalize for email comparison
        identifier_normalized = email_or_username.lower().strip()

        # Find user by email or username
        result = await db.execute(
            select(User).where(
                or_(
                    User.email == identifier_normalized,
                    User.username == email_or_username,
                )
            )
        )
        user = result.scalar_one_or_none()

        # Verify user exists and password matches
        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )

        # Check if account is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated",
            )

        # Update last login timestamp
        user.last_login_at = datetime.now(timezone.utc)
        await db.flush()

        logger.info(f"User authenticated: {user.username}")
        return user

    async def create_session(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        duration_hours: int = DEFAULT_SESSION_DURATION_HOURS,
    ) -> str:
        """Create a new session for a user.

        Generates a secure session token and stores its hash in the database.
        The plain token is returned once and should be sent to the client.

        Args:
            db: Database session
            user_id: UUID of the user
            ip_address: Client IP address (optional)
            user_agent: Client user agent string (optional)
            duration_hours: Session duration in hours (default 24)

        Returns:
            Plain session token (send to client, store as cookie/header)
        """
        # Generate token and hash
        token = generate_session_token()
        token_hash = hash_token(token)

        # Calculate expiration
        expires_at = datetime.now(timezone.utc) + timedelta(hours=duration_hours)

        # Create session record
        session = UserSession(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent[:500] if user_agent else None,
        )
        db.add(session)
        await db.flush()

        logger.info(f"Created session for user {user_id}, expires at {expires_at}")
        return token

    async def validate_session(
        self,
        db: AsyncSession,
        token: str,
    ) -> User:
        """Validate a session token and return the associated user.

        Checks that the session exists, is not expired, and the user
        is still active.

        Args:
            db: Database session
            token: Plain session token from client

        Returns:
            User associated with the session

        Raises:
            HTTPException: If session is invalid, expired, or user is inactive
        """
        token_hash = hash_token(token)
        now = datetime.now(timezone.utc)

        # Find valid session
        result = await db.execute(
            select(UserSession)
            .where(
                and_(
                    UserSession.token_hash == token_hash,
                    UserSession.expires_at > now,
                )
            )
        )
        session = result.scalar_one_or_none()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session",
            )

        # Get the user and check if active
        result = await db.execute(
            select(User).where(User.id == session.user_id)
        )
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is not active",
            )

        return user

    async def invalidate_session(
        self,
        db: AsyncSession,
        token: str,
    ) -> bool:
        """Invalidate a session (logout).

        Deletes the session from the database.

        Args:
            db: Database session
            token: Plain session token to invalidate

        Returns:
            True if session was found and deleted, False otherwise
        """
        token_hash = hash_token(token)

        # Find and delete the session
        result = await db.execute(
            select(UserSession).where(UserSession.token_hash == token_hash)
        )
        session = result.scalar_one_or_none()

        if session:
            await db.delete(session)
            await db.flush()
            logger.info(f"Invalidated session for user {session.user_id}")
            return True

        return False

    async def create_api_key(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        name: str,
        permissions: Optional[dict] = None,
        expires_at: Optional[datetime] = None,
    ) -> Tuple[str, APIKey]:
        """Create a new API key for a user.

        Generates a secure API key with optional permissions and expiration.
        The plain key is returned once and should be shown to the user.

        Args:
            db: Database session
            user_id: UUID of the user
            name: Human-readable name for the key
            permissions: Optional permissions dict (e.g., {"chat": True, "upload": False})
            expires_at: Optional expiration datetime (None = never expires)

        Returns:
            Tuple of (plain_api_key, APIKey_record)

        Raises:
            HTTPException: If an active API key with the same name already exists
        """
        # Check for duplicate name for this user
        result = await db.execute(
            select(APIKey).where(
                and_(
                    APIKey.user_id == user_id,
                    APIKey.name == name,
                    APIKey.is_active == True,
                )
            )
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An active API key with this name already exists",
            )

        # Generate key and hash
        key = generate_api_key()
        key_hash = hash_token(key)

        # Create API key record
        api_key = APIKey(
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions or {},
            expires_at=expires_at,
            is_active=True,
        )
        db.add(api_key)
        await db.flush()
        await db.refresh(api_key)

        logger.info(f"Created API key '{name}' for user {user_id}")
        return key, api_key

    async def validate_api_key(
        self,
        db: AsyncSession,
        key: str,
    ) -> Tuple[User, dict]:
        """Validate an API key and return user and permissions.

        Checks that the key exists, is active, and not expired.
        Updates last_used_at timestamp on successful validation.

        Args:
            db: Database session
            key: Plain API key from client

        Returns:
            Tuple of (User, permissions_dict)

        Raises:
            HTTPException: If key is invalid, inactive, or expired
        """
        key_hash = hash_token(key)
        now = datetime.now(timezone.utc)

        # Find valid API key
        result = await db.execute(
            select(APIKey).where(
                and_(
                    APIKey.key_hash == key_hash,
                    APIKey.is_active == True,
                )
            )
        )
        api_key = result.scalar_one_or_none()

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        # Check expiration
        if api_key.expires_at and api_key.expires_at < now:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key has expired",
            )

        # Get the user
        result = await db.execute(
            select(User).where(User.id == api_key.user_id)
        )
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is not active",
            )

        # Update last used timestamp
        api_key.last_used_at = now
        await db.flush()

        return user, api_key.permissions or {}

    async def revoke_api_key(
        self,
        db: AsyncSession,
        key_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
    ) -> bool:
        """Revoke (deactivate) an API key.

        Marks the key as inactive. Optionally verifies ownership.

        Args:
            db: Database session
            key_id: UUID of the API key to revoke
            user_id: Optional user ID to verify ownership

        Returns:
            True if key was found and revoked, False otherwise

        Raises:
            HTTPException: If user_id provided and doesn't own the key
        """
        # Build query
        query = select(APIKey).where(
            and_(
                APIKey.id == key_id,
                APIKey.is_active == True,
            )
        )

        if user_id:
            query = query.where(APIKey.user_id == user_id)

        result = await db.execute(query)
        api_key = result.scalar_one_or_none()

        if not api_key:
            if user_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="API key not found or you don't have permission to revoke it",
                )
            return False

        # Revoke the key
        api_key.is_active = False
        await db.flush()

        logger.info(f"Revoked API key {key_id}")
        return True


# =============================================================================
# Global Service Instance
# =============================================================================

auth_service = AuthService()


# =============================================================================
# FastAPI Dependencies
# =============================================================================


async def get_current_user(
    authorization: str = Header(..., description="Bearer token or API key"),
    db: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency for protected routes requiring authentication.

    Supports two authentication methods:
    1. Bearer token (session): "Bearer <session_token>"
    2. API key: "ApiKey <api_key>" or just the raw API key starting with "rag_"

    Args:
        authorization: Authorization header value
        db: Database session (injected)

    Returns:
        Authenticated User instance

    Raises:
        HTTPException: If authentication fails

    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"user": user.username}
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Parse authorization header
    auth_lower = authorization.lower()

    if auth_lower.startswith("bearer "):
        # Session token authentication
        token = authorization[7:]  # Remove "Bearer " prefix
        return await auth_service.validate_session(db, token)

    elif auth_lower.startswith("apikey "):
        # API key with prefix
        key = authorization[7:]  # Remove "ApiKey " prefix
        user, _ = await auth_service.validate_api_key(db, key)
        return user

    elif authorization.startswith(API_KEY_PREFIX):
        # Raw API key (starts with rag_)
        user, _ = await auth_service.validate_api_key(db, authorization)
        return user

    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Use 'Bearer <token>' or 'ApiKey <key>'",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    authorization: Optional[str] = Header(None, description="Optional Bearer token or API key"),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """FastAPI dependency for routes with optional authentication.

    Returns the authenticated user if valid credentials are provided,
    or None if no credentials are provided. Still raises error for
    invalid credentials.

    Args:
        authorization: Optional Authorization header value
        db: Database session (injected)

    Returns:
        Authenticated User instance or None

    Usage:
        @app.get("/public")
        async def public_route(user: Optional[User] = Depends(get_optional_user)):
            if user:
                return {"user": user.username}
            return {"user": "anonymous"}
    """
    if not authorization:
        return None

    # Use the same logic as get_current_user
    return await get_current_user(authorization=authorization, db=db)


async def get_current_user_with_permissions(
    authorization: str = Header(..., description="Bearer token or API key"),
    db: AsyncSession = Depends(get_db),
) -> Tuple[User, Optional[dict]]:
    """FastAPI dependency that returns user and permissions.

    For session authentication, permissions is None.
    For API key authentication, permissions contains the key's permissions dict.

    Args:
        authorization: Authorization header value
        db: Database session (injected)

    Returns:
        Tuple of (User, permissions_dict or None)

    Usage:
        @app.get("/protected")
        async def protected_route(
            auth: Tuple[User, Optional[dict]] = Depends(get_current_user_with_permissions)
        ):
            user, permissions = auth
            if permissions and not permissions.get("chat"):
                raise HTTPException(403, "API key doesn't have chat permission")
            return {"user": user.username}
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth_lower = authorization.lower()

    if auth_lower.startswith("bearer "):
        token = authorization[7:]
        user = await auth_service.validate_session(db, token)
        return user, None

    elif auth_lower.startswith("apikey "):
        key = authorization[7:]
        return await auth_service.validate_api_key(db, key)

    elif authorization.startswith(API_KEY_PREFIX):
        return await auth_service.validate_api_key(db, authorization)

    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format",
            headers={"WWW-Authenticate": "Bearer"},
        )
