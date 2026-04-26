"""Add knowledge_pool to documents.

Revision ID: 0002
Revises: 0001
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "knowledge_pool",
            sa.String(32),
            nullable=False,
            server_default="voice",
        ),
    )


def downgrade() -> None:
    op.drop_column("documents", "knowledge_pool")
