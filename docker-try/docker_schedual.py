# 容器的调度方式
import docker
from collections import deque

client = docker.from_env()
MIN_PORT = 5001
MAX_PORT = 5009
ALL_PORTS = [i for i in range(MIN_PORT, MAX_PORT + 1)]


def getWarmQueue():
    warm_list = client.containers.list()
    warm_list = [i for i in warm_list if "hanwen_torch" in i.name]
    return deque(warm_list)


# 获取详细的信息，但是好像这个函数并不重要
def getWarmContainersDetail(warm_list):
    # {'container': <Container: 0a37d32b44>, 'name': 'hanwen_torch_6', 'status': 'running', 'port': 5006}
    return [{"container": i,
             "name": i.name,
             "status": i.status,
             "port": int(i.ports['5000/tcp'][0]['HostPort'])} for i in warm_list
            if "hanwen_torch" in i.name]


import urllib.request
import json
import time

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
    print("container_{}".format(container_port), content["result"], content["possibility"], "total_time: ",
          content["total_time"])
    return content


imgUrlList = ["https://dd-static.jd.com/ddimg/jfs/t1/162905/7/17846/76460/6075526dE6af30fb3/b8ed4431cb5c628a.jpg",
              "https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg"]

imgUrlQueue = deque(imgUrlList * 20)

from os_docker import create_new_container


def getFeasiblePort(warm_list, all_ports):
    used_ports = [int(i.ports['5000/tcp'][0]['HostPort']) for i in warm_list]
    feasible_ports = [i % 5000 for i in all_ports if i not in used_ports]
    print(feasible_ports)

    return (True, feasible_ports[0]) if len(feasible_ports) > 0 else (False, -1)


warm_list = client.containers.list()
warmQueue = getWarmQueue()
# if len(imgUrlQueue) > 5 * len(warmQueue):
#     flag, feasible = getFeasible(warm_list, ALL_PORTS)
#     print(feasible)
#     if flag: create_new_container(feasible)


while imgUrlQueue:
    imgUrl = imgUrlQueue.popleft()
    if len(warmQueue) == 0: warmQueue = getWarmQueue()
    worker = warmQueue.popleft()
    port = worker.ports['5000/tcp'][0]['HostPort']%5000
    result = getPredictResults(port, imgUrl)
