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

    # print("container_{}".format(container_port), content["result"], content["possibility"], "total_time: ",
    #       content["total_time"])
    return content


def getWarmQueue():
    warm_list = client.containers.list()
    warm_list = [i for i in warm_list if "hanwen_torch" in i.name]
    return deque(warm_list)


if __name__ == '__main__':
    time0 = time.time()
    warmQueue = getWarmQueue()

    while redis_conn.llen(target):
        # details = json.loads(redis_conn.rpop(target))
        details = json.loads(redis_conn.brpop(target)[1])


        if len(warmQueue) == 0: warmQueue = getWarmQueue()
        worker = warmQueue.popleft()
        port = worker.ports['5000/tcp'][0]['HostPort']
        result = getPredictResults(port, imgUrl=details["url"])
        details["result"] = result["result"]
        print(details["idx"], time.time() - time0)

        # redis_conn2.lpush(redis_result_list, json.dumps(details))

    print("ALL Time:", time.time() - time0)
