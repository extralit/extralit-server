import asyncio


async def astreamer(generator):
    try:
        for i in generator:
            yield i
            await asyncio.sleep(.01)
    except asyncio.CancelledError as e:
        print('cancelled')
    except Exception as e:
        print(e)
