import redis
import json
import time

redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379)
redis_conn = redis.Redis(connection_pool=redis_pool)
redis_conn2 = redis.Redis(connection_pool=redis_pool)

target = "mylist"
redis_result_list = "redis_result"

while redis_conn.llen(target):
    result=json.loads(redis_conn.rpop(target))
    print(result)

while redis_conn2.llen(redis_result_list):
    result=json.loads(redis_conn2.rpop(redis_result_list))
    print(result)
