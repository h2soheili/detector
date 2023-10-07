from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import (Integer, BigInteger, DateTime, func)
from sqlalchemy import (String)
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.orm import mapped_column

from backend.config import settings
from backend.global_log import structlog

logger = structlog.get_logger(__name__)

# from sqlalchemy import event
# from sqlalchemy.engine import Engine
# import time
#
#
# @event.listens_for(Engine, "before_cursor_execute")
# def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
#     conn.info.setdefault("query_start_time", []).append(time.time())
#     # logger.debug("Start Query: ", statement=statement)
#
#
# @event.listens_for(Engine, "after_cursor_execute")
# def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
#     total = time.time() - conn.info["query_start_time"].pop(-1)
#     logger.debug(f"Query Complete! Total Time: {total}", statement=statement)


async_engines_dict = dict()


def async_db_engine(pool_size=2):
    if async_engines_dict.get(pool_size):
        return async_engines_dict.get(pool_size)
    eng = create_async_engine(settings.SQL_DSN, pool_pre_ping=True, pool_size=pool_size, echo=False, )
    async_engines_dict[pool_size] = eng
    return eng


def async_db_session(pool_size=2) -> AsyncSession:
    try:
        engine = async_db_engine(pool_size)
        async_session = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)()
        return async_session
    except Exception as e:
        print(e)


engines_dict = dict()


def db_engine(pool_size=2):
    if engines_dict.get(pool_size):
        return engines_dict.get(pool_size)
    eng = create_engine(settings.SQL_DSN, pool_pre_ping=True,
                        pool_recycle=10000, pool_size=pool_size)
    engines_dict[pool_size] = eng
    return eng


def db_session(pool_size=2) -> Session:
    try:
        engine = db_engine(pool_size)
        session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))()
        return session
    except Exception as e:
        print(e)


class Serializer(object):

    def serialize(self):
        serialized = {}
        for c in inspect(self).attrs.keys():
            serialized[c] = getattr(self, c)
        return serialized

    @staticmethod
    def serialize_list(rows: List[Any]):
        return [r.serialize() for r in rows]


class Base(AsyncAttrs, DeclarativeBase, Serializer):
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True, autoincrement=True)
    __name__: str
    __table_args__ = {"extend_existing": True}

    date_created: Mapped[Optional[datetime]] = mapped_column(DateTime, default=func.current_timestamp(),
                                                             nullable=True)
    date_modified: Mapped[Optional[datetime]] = mapped_column(DateTime, default=func.current_timestamp(),
                                                              onupdate=func.current_timestamp(), nullable=True, )
    date_archived: Mapped[Optional[Any]] = mapped_column(DateTime, default=None, nullable=True)

    __mapper_args__ = {"eager_defaults": True}

    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()


class User(Base):
    __tablename__ = 'user'
    key: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    username: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    password: Mapped[Optional[str]] = mapped_column(String(180), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    state: Mapped[int] = mapped_column(Integer, nullable=True)
