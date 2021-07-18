# 高并发测试结果
import urllib.request
import json
import time

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) \
        AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/35.0.1916.114 Safari/537.36',
    'Cookie': 'AspxAutoDetectCookieSupport=1'
}


def getPredictResults(container, imgUrl):
    baseUrl = "http://127.0.0.1:500{}/url/".format(container)
    targetUrl = baseUrl + imgUrl
    req = urllib.request.Request(url=targetUrl, headers=header)
    response = urllib.request.urlopen(req)
    content = response.read()
    content = json.loads(content)
    response.close()
    print("c{}".format(container), content["result"], content["possibility"], "total_time: ", content["total_time"])
    return content


imgUrlList = []
imgUrl = "https://dd-static.jd.com/ddimg/jfs/t1/162905/7/17846/76460/6075526dE6af30fb3/b8ed4431cb5c628a.jpg"
imgUrlList.append(imgUrl)
imgUrl = "https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg"
imgUrlList.append(imgUrl)

getPredictResults(3, imgUrl)
getPredictResults(2, imgUrl)
getPredictResults(1, imgUrl)

for i in range(10):
    for j in [1, 2, 3]:
        for imgUrl in imgUrlList:
            time0=time.time()
            getPredictResults(j, imgUrl)
            print(time.time()-time0,"\n")
