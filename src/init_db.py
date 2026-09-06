import asyncio
import uuid
from sqlalchemy import select, update
from api import Base, DBUser, engine, AsyncSessionLocal
from src.auth import hash_password


async def init_db():
    print("Initializing Nydra Database & Security...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(DBUser).where(DBUser.username == "guest"))
        guest = result.scalar_one_or_none()

        if guest:
            print("Updating existing 'guest' user...")
            await db.execute(
                update(DBUser)
                .where(DBUser.username == "guest")
                .values(hashed_password=hash_password("password"), is_admin=False)
            )
        else:
            print("Creating new 'guest' user (non-admin)...")
            db.add(
                DBUser(
                    id=str(uuid.uuid4()),
                    username="guest",
                    email="guest@nydra.ai",
                    hashed_password=hash_password("password"),
                    is_active=True,
                    is_admin=False,
                    settings={},
                )
            )

        await db.commit()

    print("Database ready. Guest user (non-admin): guest / password")


if __name__ == "__main__":
    asyncio.run(init_db())
