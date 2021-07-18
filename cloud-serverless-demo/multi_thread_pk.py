import threading
import time
import json
from redis_consumer import getPredictResults
from redis_consumer import redis_conn, redis_conn2

exitFlag = 0
target = "mylist"
redis_result_list = "redis_result"


class myThread(threading.Thread):
    def __init__(self, threadID, name, port):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.port = port

    def run(self):
        print("开始线程：" + self.name)
        while redis_conn.llen(target):
            details = json.loads(redis_conn.brpop(target)[1])
            print(self.name, details["idx"], time.time() - time0)
            result = getPredictResults("http://47.102.148.216", self.port, details["url"])
        print("退出线程：" + self.name)


# 创建新线程
thread1 = myThread(1, "Thread-1", 5000)
thread2 = myThread(2, "Thread-2", 5001)
thread3 = myThread(3, "Thread-3", 5002)
thread4 = myThread(4, "Thread-4", 5003)
thread5 = myThread(5, "Thread-5", 5004)

import time

time0 = time.time()
# 开启新线程
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread1.join()
thread2.join()
thread3.join()
thread4.join()
thread5.join()
print("ALL Time:", time.time() - time0)
print("退出主线程")
