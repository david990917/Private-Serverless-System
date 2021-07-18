import redis
import json
import docker
import time
import urllib.request
from collections import deque

client = docker.from_env()
redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379)
redis_conn = redis.Redis(connection_pool=redis_pool)
redis_conn2 = redis.Redis(connection_pool=redis_pool)

target = 'mylist'
redis_result_list = "redis_result"


def batch_lpop(n, target=target):
    pipe = redis_conn.pipeline()
    pipe.lrange(target, 0, n - 1)
    pipe.ltrim(target, n, -1)
    # 这里的 data 还是完整的信息，但是对于我们不需要了
    # data = pipe.execute()
    # 现在返回的是一个 需要json.loads 就可以的 数组
    data = pipe.execute()[0]
    return data


header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) \
        AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/35.0.1916.114 Safari/537.36',
    'Cookie': 'AspxAutoDetectCookieSupport=1'
}


def getPredictResults(container_port, imgUrl):
    baseUrl = "http://127.0.0.1:{}/url/".format(container_port)
    targetUrl = baseUrl + imgUrl
    req = urllib.request.Request(url=targetUrl, headers=header)
    response = urllib.request.urlopen(req)
    content = response.read()
    content = json.loads(content)
    response.close()
    return content


def getWarmQueue():
    warm_list = client.containers.list()
    warm_list = [i for i in warm_list if "hanwen_torch" in i.name]
    return deque(warm_list)


import time
import asyncio
from aiohttp import ClientSession

tasks = []


async def hanwen_async_predict(url):
    async with ClientSession() as session:
        async with session.get(url) as response:
            print('Hello World:%s' % time.time())
            return await response.read()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    time0 = time.time()
    while redis_conn.llen(target):
        redis_length = redis_conn.llen(target)
        warmQueue = getWarmQueue()
        worker_num = len(warmQueue)
        worker_port = [i.ports['5000/tcp'][0]['HostPort'] for i in warmQueue]
        worker_url = ["http://127.0.0.1:{}/url/".format(i) for i in worker_port]

        redis_fetch_batch = batch_lpop(min(redis_length, worker_num))
        redis_fetch_batch_predict = [json.loads(i) for i in redis_fetch_batch]

        tasks = []
        for idx in range(min(redis_length, worker_num)):
            request = redis_fetch_batch_predict[idx]
            imgURL = request["url"]
            workerURL = worker_url[idx]
            predictUrl = workerURL + imgURL
            task = asyncio.ensure_future(hanwen_async_predict(predictUrl))
            tasks.append(task)
        batchResult = loop.run_until_complete(asyncio.gather(*tasks))
        for i in batchResult:
            currTarget = json.loads(i)
            print(currTarget["time"]["photo_time"])
        print("\n")

    print("ALL Time:", time.time() - time0)
