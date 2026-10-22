"""add customer active flag

Revision ID: a3c1e5f7b9d2
Revises: 57f9b31937ca
Create Date: 2026-10-22 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3c1e5f7b9d2'
down_revision: Union[str, Sequence[str], None] = '57f9b31937ca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('customers', sa.Column('active', sa.Boolean(), server_default=sa.true(), nullable=False))
    op.create_index(op.f('ix_customers_active'), 'customers', ['active'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_customers_active'), table_name='customers')
    op.drop_column('customers', 'active')
