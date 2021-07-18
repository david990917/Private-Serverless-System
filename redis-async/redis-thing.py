# Redis 的使用 - 连续弹出多个
import redis
import json
import time

redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379)
redis_conn = redis.Redis(connection_pool=redis_pool)

target = "mylist"


def batch_lpop(n, target=target):
    pipe = redis_conn.pipeline()
    pipe.lrange(target, 0, n - 1)
    pipe.ltrim(target, n, -1)
    data = pipe.execute()
    return data


a = batch_lpop(5, target="mylist")
print(a[1]) # 永远是 True 即便没有东西了
for i in range(len(a[0])):
    print(a[0][i])
    print(json.loads(a[0][i]))