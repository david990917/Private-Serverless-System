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


def getPredictResults(base, container_port, imgUrl):
    baseUrl = "{}:{}/url/".format(base, container_port)
    targetUrl = baseUrl + imgUrl
    req = urllib.request.Request(url=targetUrl, headers=header)
    response = urllib.request.urlopen(req)
    content = response.read()
    content = json.loads(content)
    response.close()
    return content


warmQueue = ["http://47.102.148.216", "http://47.102.210.246"]
worker_num = len(warmQueue)

if __name__ == '__main__':
    time0 = time.time()
    workerID = 0
    while redis_conn.llen(target):
        details = json.loads(redis_conn.brpop(target)[1])
        print(details["idx"], time.time() - time0)

        worker = warmQueue[workerID]  # 现在是机器ip
        result = getPredictResults(worker, 5000, imgUrl=details["url"])
        details["result"] = result["result"]
        workerID = (workerID + 1) % worker_num

    print("ALL Time:", time.time() - time0)
