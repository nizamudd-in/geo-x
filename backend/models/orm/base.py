"""
GeoSense-X — SQLAlchemy Declarative Base & Shared Mixins
==========================================================
All ORM models inherit from Base defined here.
TimestampMixin adds created_at / updated_at to any model automatically.

Naming conventions are configured so Alembic generates consistent
constraint names across databases.

Usage:
    from models.orm.base import Base, TimestampMixin
    class MyModel(TimestampMixin, Base):
        __tablename__ = "my_table"
        id: Mapped[int] = mapped_column(primary_key=True)
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Naming conventions ensure Alembic generates deterministic constraint names
# across all supported databases (PostgreSQL, SQLite for tests).
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base for all GeoSense-X ORM models.

    Configures metadata with naming conventions so Alembic generates
    deterministic migration names. All models inherit from this class.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """
    Mixin that adds created_at and updated_at columns to any model.

    created_at: Set once at INSERT time (server default).
    updated_at: Updated automatically on every UPDATE via onupdate trigger.

    Inheriting models should list TimestampMixin before Base in MRO:
        class MyModel(TimestampMixin, Base):
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Timestamp when this row was first inserted (UTC).",
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
        doc="Timestamp of the most recent update to this row (UTC).",
    )
