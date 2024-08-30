from zep_cloud.client import AsyncZep
from dotenv import load_dotenv
import os
from uuid import uuid4


async def add_new_user(
    zep_client: AsyncZep, user_id: str, email, first_name, last_name, metadata: dict
):
    new_user = await zep_client.user.add(
        user_id=user_id,
        email=email,
        first_name=first_name,
        last_name=last_name,
        metadata=metadata,
    )
    return new_user


async def check_user_is_exists(zep_client: AsyncZep, user_id):
    try:
        await zep_client.user.get(user_id=user_id)
        return True
    except Exception as e:
        # User does not exist
        return False


async def main(user_id=None):

    API_KEY = os.getenv("ZEP_API_KEY")
    client = AsyncZep(api_key=API_KEY)
    if await check_user_is_exists(client, user_id):
        user = await client.user.get(user_id=user_id)
    else:
        user = await add_new_user(
            user_id=user_id,
            email="123123@example.com",
            first_name="Tenison",
            last_name="Teng",
            metadata={"user_name": "tension", "age": 22, "gender": "male"},
        )

    print(user)


if __name__ == "__main__":
    import asyncio

    load_dotenv()
    user_id = str(uuid4().hex)

    asyncio.run(main(user_id="test1"))
