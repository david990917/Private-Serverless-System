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

mapping = {}
with open("url_idx.txt", "r") as f:
    details = [s.strip() for s in f.readlines()]
    for idx, url in enumerate(details):
        mapping[url] = idx

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) \
        AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/35.0.1916.114 Safari/537.36',
    'Cookie': 'AspxAutoDetectCookieSupport=1'
}


def download_imgs(imgUrl):
    time0 = time.time()
    # 下载图片到指定的路径
    idx = mapping[imgUrl]
    filename = "/Users/starky/PycharmProjects/docker-torch/storage/hanwen_torch_0/{}.jpg".format(idx)
    # filename = "/Users/starky/PycharmProjects/docker-torch/test-torch-http-for-copy/exec/{}.jpg".format(idx)
    print(filename)
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    download_single_time = time.time() - time0
    return download_single_time


def getWarmQueue():
    warm_list = client.containers.list()
    warm_list = [i for i in warm_list if "hanwen_torch" in i.name]
    return deque(warm_list)


if __name__ == '__main__':
    target = "mylistCOPY"
    count = 0
    flag=0
    download_copy_time = 0
    while not flag or redis_conn.llen(target):
        if redis_conn.llen(target) and not flag:
            print("开始处理下载了", time.time())
            flag=True
        details = json.loads(redis_conn.blpop(target)[1])
        redis_conn.rpush("mylist", json.dumps(details))
        url = details["url"].strip()
        count+=1
        time0 = time.time()
        download_imgs(url)
        download_copy_time += time.time() - time0
    print(download_copy_time/count)
