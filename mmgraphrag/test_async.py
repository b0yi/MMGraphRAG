import asyncio

async def corroutine_a():
    print("Coroutine A is running...")
    await asyncio.sleep(1)
    print("Coroutine A is done.")

async def corroutine_b():
    print("Coroutine B is running...")
    await asyncio.sleep(1)
    print("Coroutine B is done.")

async def main():
    task1 = asyncio.create_task(corroutine_a())
    task2 = asyncio.create_task(corroutine_b())
    result1 = await task1
    result2 = await task2

if __name__ == "__main__":
    asyncio.run(main())