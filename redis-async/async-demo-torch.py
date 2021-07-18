import time
import asyncio
from aiohttp import ClientSession
import json

url = "http://127.0.0.1:{}/url/" \
      + "https://dd-static.jd.com/ddimg/jfs/t1/162905/7/17846/76460/6075526dE6af30fb3/b8ed4431cb5c628a.jpg"


async def hello(url):
    async with ClientSession() as session:
        async with session.get(url) as response:
            print('Hello World:%s' % time.time())
            return await response.read()


def run():
    tasks = []
    for i in range(1):
        task = asyncio.ensure_future(hello(url.format(i + 5000)))
        tasks.append(task)
    result = loop.run_until_complete(asyncio.gather(*tasks))
    for i in result:
        target=json.loads(i)
        print(target["time"]["photo_time"])


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    for i in range(100):
        print("\n{}\n".format(i))
        run()
