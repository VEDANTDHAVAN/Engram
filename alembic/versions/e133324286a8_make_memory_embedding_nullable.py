"""make memory embedding nullable

Revision ID: e133324286a8
Revises: df464ed91513
Create Date: 2026-08-14 20:54:05.270018

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e133324286a8'
down_revision: Union[str, Sequence[str], None] = 'df464ed91513'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
