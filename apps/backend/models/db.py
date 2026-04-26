"""Async SQLAlchemy engine + session factory.

We keep the engine lazy so that importing the module never opens a
connection. Tests can override `get_session` via FastAPI dependency
overrides without touching the global engine.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from apps.backend.config import get_settings
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base shared by every ORM entity."""


_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine():
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


def get_engine():
    global _engine, _session_factory
    if _engine is None:
        _engine = _build_engine()
        _session_factory = async_sessionmaker(
            _engine, expire_on_commit=False, class_=AsyncSession
        )
    return _engine


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    get_engine()
    assert _session_factory is not None
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency producing an `AsyncSession`."""
    get_engine()
    assert _session_factory is not None
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create tables on startup. For real deployments use Alembic instead."""
    from apps.backend.models import entities  # noqa: F401  (register mappers)

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("database_initialised")
