from datetime import datetime
from typing import Any, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel
from sqlalchemy import text, and_, Sequence
from sqlalchemy.future import select
from sqlalchemy.orm import class_mapper

from backend.db import (Base, async_db_session, User)
from backend.global_log import structlog
from backend.schemas import (UserCreateDTO, UserUpdateDTO, UserInDB)

ModelType = TypeVar("ModelType", bound=Base)
SchemaCreateType = TypeVar("SchemaCreateType", bound=BaseModel)
SchemaUpdateType = TypeVar("SchemaUpdateType", bound=BaseModel)
SchemaTypeInDB = TypeVar("SchemaTypeInDB", bound=BaseModel)

logger = structlog.get_logger(__name__)


class CRUDBase(Generic[ModelType, SchemaCreateType, SchemaUpdateType, SchemaTypeInDB]):
    def __init__(self, model: Type[ModelType]):
        """
        CRUD object with default methods to Create, Read, Update, Delete (CRUD).

        **Parameters**

        * `model`: A SQLAlchemy model class
        * `schema`: A Pydantic model (schema) class
        """
        self.model = model

    def with_nolock(self, query):
        # query = query.with_hint(self.model, 'WITH (NOLOCK)')
        # query = query.with_lockmode("read_nowait")
        return query

    async def get(self, id: Any, only_actives: bool = True) -> Optional[ModelType]:
        async with async_db_session() as session:
            query = select(self.model)
            query = query.where(self.model.id == id)
            if only_actives:
                query = query.where(self.model.state == 1)
            query = self.with_nolock(query)
            query = await session.execute(query)
            return query.scalars().first()

    async def get_with_key(self, key: Any, only_actives: bool = True) -> Optional[ModelType]:
        async with async_db_session() as session:
            query = select(self.model)
            query = query.where(self.model.key == key)
            if only_actives:
                query = query.where(self.model.state == 1)
            query = self.with_nolock(query)
            query = await session.execute(query)
            return query.scalars().first()

    def filter_only_actives(self, query, only_actives: bool = True):
        if only_actives:
            query = query.where(self.model.state == 1)
        return query

    def filter_by_user_id(self, query, user_id: int = None):
        if user_id is not None and vars(self.model).get("time"):
            query = query.filter(self.model.user_id == user_id)
        return query

    def filter_by_time(self, query,
                       after: datetime = None,
                       before: datetime = None, ):
        if (after or before) and vars(self.model).get("time"):
            if after and before:
                query = query.filter(and_(self.model.time >= after, self.model.time <= before))
            else:
                if after:
                    query = query.where(self.model.time >= after)
                elif before:
                    query = query.where(self.model.time <= before)
        return query

    async def get_multi(self,
                        key: Any = None,
                        id: Any = None,
                        after: datetime = None,
                        before: datetime = None,
                        time: datetime = None,
                        skip: int = 0,
                        limit: int = 100,
                        desc: bool = True,
                        only_actives: bool = True,
                        user_id: int = None
                        ) -> Sequence[ModelType]:

        async with async_db_session() as session:
            query = select(self.model)
            query = self.filter_only_actives(query, only_actives)
            query = self.filter_by_time(query, after, before)
            query = self.filter_by_user_id(query, user_id)
            order_by = "desc" if desc else "asc"
            query = query.order_by(text(f"id {order_by}"))
            query = query.offset(skip).limit(limit)
            query = self.with_nolock(query)
            query = await session.execute(query)
            query = query.scalars().all()
            return query

    async def create(self, db_obj: SchemaCreateType) -> Optional[ModelType]:
        async with async_db_session() as session:
            try:
                session.add(db_obj)
                await session.commit()
                await session.refresh(db_obj)
                return db_obj
            except Exception as error:
                await session.rollback()
                logger.error("model create error", error=error, db_obj=db_obj)
                raise error
                # raise HTTPException(
                #     status_code=500,
                #     detail=str(error)
                # )

    async def update(self, db_obj: ModelType, obj_in: SchemaUpdateType) -> ModelType:
        async with async_db_session() as session:

            try:
                # print("update")
                # print(db_obj)
                # print(obj_in)
                # db_obj = jsonable_encoder(db_obj)
                # obj_in = obj_in.dict()
                # obj_data = db_obj.__dict__
                # obj_data = obj_in
                update_data = {}
                if isinstance(obj_in, dict):
                    update_data = obj_in
                else:
                    try:
                        update_data = obj_in.dict(exclude_unset=True)
                    except Exception as e:
                        try:
                            update_data = obj_in.__dict__
                            # if "_sa_instance_state" in update_data:
                            #     update_data.pop("_sa_instance_state")
                        except Exception as ee:
                            pass

                # update_data = jsonable_encoder(update_data)

                # if "id" in update_data:
                #     update_data.pop("id")
                # for field in obj_data:
                #     if field in update_data:
                #         setattr(db_obj, field, update_data[field])
                # print('update_data')
                # print(update_data)
                for field in update_data:
                    if field not in ("_sa_instance_state", "id"):
                        setattr(db_obj, field, update_data[field])
                # db_obj = self.model(**db_obj)
                # print('db_obj', db_obj)
                # print(db_obj.serialize())
                session.add(db_obj)
                await session.commit()
                await session.refresh(db_obj)
                return db_obj
            except Exception as error:
                await session.rollback()
                logger.error("model update error", error=error, db_obj=db_obj.serialize(), obj_in=obj_in)
                raise error
                # raise HTTPException(
                #     status_code=500,
                #     detail=str(error)
                # )

    async def remove(self, id: Any) -> Optional[Any]:
        async with async_db_session() as session:
            try:
                obj = await session.execute(select(self.model).get(id))
                await session.delete(obj)
                await session.commit()
                return id
            except Exception as error:
                await session.rollback()
                logger.error("model remove error", error=error)
                raise error
                # raise HTTPException(
                #     status_code=500,
                #     detail=str(error)
                # )

    async def get_by_time(self, time: datetime, key: Any = None) -> Optional[ModelType]:
        async with async_db_session() as session:
            query = select(self.model)
            query = query.where(self.model.time == time).first()
            query = self.with_nolock(query)
            query = await session.execute(query)
            query = query.scalars().first()
            return query

    @staticmethod
    def object_to_dict(obj: ModelType, found=None) -> dict:
        if found is None:
            found = set()
        mapper = class_mapper(obj.__class__)
        columns = [column.key for column in mapper.columns]
        get_key_value = lambda c: (c, getattr(obj, c).isoformat()) if isinstance(getattr(obj, c), datetime) else (
            c, getattr(obj, c))
        out = dict(map(get_key_value, columns))
        for name, relation in mapper.relationships.items():
            if relation not in found:
                found.add(relation)
                related_obj = getattr(obj, name)
                if related_obj is not None:
                    if relation.uselist:
                        out[name] = [CRUDBase.object_to_dict(child, found) for child in related_obj]
                    else:
                        out[name] = CRUDBase.object_to_dict(related_obj, found)
        return out

    @staticmethod
    def list_of_object_to_list_of_dict(rows: List[ModelType], ) -> List[dict]:
        return [CRUDBase.object_to_dict(row) for row in rows]


class CRUDUser(CRUDBase[User, UserCreateDTO, UserUpdateDTO, UserInDB]):
    async def create(self, obj_in) -> Any:
        async with async_db_session() as session:
            db_user = await self.get_by_username(username=obj_in.username)
            if not db_user:
                db_user = await super().create(obj_in)
                await session.commit()
            return db_user

    async def get_by_username(self, username: str) -> Optional[User]:
        async with async_db_session() as session:
            query = select(self.model)
            query = query.where(User.username == username)
            query = self.with_nolock(query)
            query = await session.execute(query)
            query = query.scalars().first()
            return query

    async def authenticate(self, username: str, password: str) -> Optional[User]:
        user = await self.get_by_username(username=username)
        # if not user:
        #     return None
        # if not verify_password(password, user.password):
        #     return None
        return user

    def is_active(self, user: User) -> bool:
        return user.state != 0

    def is_superuser(self, user: User) -> bool:
        return True


crud_user = CRUDUser(User)
