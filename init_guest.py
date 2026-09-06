
import asyncio
from api import get_db, DBUser, pwd_ctx
from sqlalchemy import select, insert

async def create_guest():
    async for db in get_db():
        # Check if user already exists
        result = await db.execute(select(DBUser).where(DBUser.username == "guest"))
        if result.scalar_one_or_none():
            print("Guest user already exists.")
            return

        # Insert guest user
        await db.execute(
            insert(DBUser).values(
                id="guest-uuid-" + str(hash("guest")),
                username="guest",
                email="guest@nydra.ai",
                hashed_password=pwd_ctx.hash("password"),
                is_active=True
            )
        )
        await db.commit()
        print("Guest user created successfully (guest / password)")
        break

if __name__ == "__main__":
    asyncio.run(create_guest())
