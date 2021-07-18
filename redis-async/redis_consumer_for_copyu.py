# copy 问题的基础测试
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
    print(time0)
    warmQueue = getWarmQueue()
    time_preprocess = 0
    time_download = 0
    time_inference = 0
    count = 0
    while redis_conn.llen(target):
        details = json.loads(redis_conn.brpop(target)[1])
        port = 5000
        result = getPredictResults(port, imgUrl=details["url"])
        details["result"] = result["result"]
        time_download += result["time"]["photo_time"]["downloading_photo_time"]
        time_preprocess += result["time"]["photo_time"]["processing_photo_time"]
        time_inference += result["time"]["photo_time"]["inference_time"]
        count += 1
        print(count, details["idx"], time.time() - time0)

    print("ALL Time:", time.time() - time0)
    print("avg_down", time_download / count)
    print("avg_pre", time_preprocess / count)
    print("avg_inf", time_inference/count)
