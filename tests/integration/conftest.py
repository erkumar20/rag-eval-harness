"""Shared fixtures for integration tests, which write to the real local Postgres (no mocks
-- see module docstrings in test_storage.py / test_runner.py for why).

Every fixture in this file that touches storage is scoped to a unique, uuid-suffixed
pipeline name and deletes everything it created on teardown. Without this, tests re-run
against the same persistent database accumulate rows across runs -- concretely, this bit
during Phase 9 development: a baseline set by one test run made a later run's "no baseline
yet" assertion fail, because the previous run's baseline row was still sitting in the table.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import delete

from eval_harness.storage.db import session_scope
from eval_harness.storage.models import Baseline, PipelineVersion, Result, Run


@pytest.fixture
def pipeline_name() -> Iterator[str]:
    """A pipeline name unique to this test invocation, cleaned up (and everything stored
    under it) after the test finishes -- so tests never see leftover state from a previous
    run and never leave permanent rows in the dev database."""
    name = f"test-{uuid.uuid4().hex[:12]}"
    yield name

    with session_scope() as session:
        version_ids = [
            row.id
            for row in session.query(PipelineVersion.id).filter(PipelineVersion.name == name)
        ]
        if version_ids:
            run_ids = [
                row.id for row in session.query(Run.id).filter(Run.pipeline_version_id.in_(version_ids))
            ]
            if run_ids:
                session.execute(delete(Result).where(Result.run_id.in_(run_ids)))
                session.execute(delete(Run).where(Run.id.in_(run_ids)))
            session.execute(delete(PipelineVersion).where(PipelineVersion.id.in_(version_ids)))
        session.execute(delete(Baseline).where(Baseline.pipeline_name == name))
