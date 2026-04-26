"""Speech analytics: criteria sets + criteria.

Revision ID: 0003
Revises: 0002
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analytics_criteria_sets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("locale", sa.String(8), nullable=False, server_default="ru"),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_table(
        "analytics_criteria",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "set_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("analytics_criteria_sets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code", sa.String(64), nullable=False),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1"),
        sa.Column("max_score", sa.Float(), nullable=False, server_default="10"),
        sa.Column("rubric", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_analytics_criteria_set_id", "analytics_criteria", ["set_id"])
    op.create_unique_constraint(
        "uq_analytics_criteria_set_code", "analytics_criteria", ["set_id", "code"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_analytics_criteria_set_code", "analytics_criteria", type_="unique")
    op.drop_index("ix_analytics_criteria_set_id", table_name="analytics_criteria")
    op.drop_table("analytics_criteria")
    op.drop_table("analytics_criteria_sets")
