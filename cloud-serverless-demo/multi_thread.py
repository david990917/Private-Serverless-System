import threading
import time
import json
from redis_consumer import getPredictResults
from redis_consumer import redis_conn, redis_conn2

exitFlag = 0
target = "mylist"


class myThread(threading.Thread):
    def __init__(self, threadID, name, host):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.host = host

    def run(self):
        print("开始线程：" + self.name)
        while redis_conn.llen(target):
            details = json.loads(redis_conn.brpop(target)[1])
            print(self.name, details["idx"], time.time() - time0)
            result = getPredictResults(self.host, 5000,details["url"])

        print("退出线程：" + self.name)


# 创建新线程
thread1 = myThread(1, "Thread-1", "http://47.102.148.216")
thread2 = myThread(2, "Thread-2", "http://47.102.210.246")


import time

time0 = time.time()
# 开启新线程
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print("ALL Time:", time.time() - time0)
print("退出主线程")
