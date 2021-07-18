import redis
import json
import time

redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379)
redis_conn = redis.Redis(connection_pool=redis_pool)

target = "mylist"

# 使用 本地的 url
count = 0
with open("/Users/starky/PycharmProjects/docker-torch/cat_1000.txt") as f:
    lines = f.readlines()
    for line in lines:
        if count==250:break
        details = {"url": line,
                   "idx": count,
                   "in_time": time.time(),
                   "result": None}
        v = redis_conn.lpush(target, json.dumps(details))
        count += 1

with open("/Users/starky/PycharmProjects/docker-torch/dog_1000.txt") as f:
    lines = f.readlines()
    for line in lines:
        if count==500:break
        details = {"url": line,
                   "idx": count,
                   "in_time": time.time(),
                   "result": None}
        v = redis_conn.lpush(target, json.dumps(details))
        count += 1
