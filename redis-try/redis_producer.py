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
        details = {"url": line,
                   "idx": count,
                   "in_time": time.time(),
                   "result": None}
        v = redis_conn.lpush(target, json.dumps(details))
        count += 1

with open("/Users/starky/PycharmProjects/docker-torch/dog_1000.txt") as f:
    lines = f.readlines()
    for line in lines:
        details = {"url": line,
                   "idx": count,
                   "in_time": time.time(),
                   "result": None}
        v = redis_conn.lpush(target, json.dumps(details))
        count += 1

# UrlList = ["https://dd-static.jd.com/ddimg/jfs/t1/162905/7/17846/76460/6075526dE6af30fb3/b8ed4431cb5c628a.jpg",
#            "https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg"]
#
#
# # UrlList的添加
# count = 0
# for i in range(100):
#     details = {"url": UrlList[0],
#                "idx": count,
#                "in_time": time.time(),
#                "result": None}
#     v = redis_conn.lpush(target, json.dumps(details))
#     count += 1
#
#     details = {"url": UrlList[1],
#                "idx": count,
#                "in_time": time.time(),
#                "result": None}
#     v = redis_conn.lpush(target, json.dumps(details))
#     count += 1
#     print(v)


# github 的链接太慢了
# count = 0
# for i in range(100):
#     details = {"url": "https://raw.githubusercontent.com/david990917/hanwenpicbed/main/Cat/{}.jpg".format(i),
#                "idx": count,
#                "in_time": time.time(),
#                "result": None}
#     v = redis_conn.lpush(target, json.dumps(details))
#     count += 1
