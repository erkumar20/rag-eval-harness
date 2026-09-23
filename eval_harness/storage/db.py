from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from eval_harness.config import get_settings


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Commits on clean exit, rolls back on exception -- the shape every repository
    function (Phase 8 step 5) and the runner (Phase 9) call into."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
