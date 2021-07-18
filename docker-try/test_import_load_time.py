# 主要是提取 import 和 load_model 的时间
import urllib.request
import json

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) \
        AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/35.0.1916.114 Safari/537.36',
    'Cookie': 'AspxAutoDetectCookieSupport=1'
}


def request_importing_time(i):
    targetUrl = "http://127.0.0.1:500{}/url/https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg".format(
        i)
    req = urllib.request.Request(url=targetUrl, headers=header)
    response = urllib.request.urlopen(req)
    content = response.read()
    content = json.loads(content)
    target = content["time"]["import_time"]
    response.close()
    # return (content["time"]["import_time"]["total_import_time"], content["time"]["loading_model_time"])
    return (target["total_import_time"], target["loading_torch_time"], target["loading_torchvision_time"],
            target["loading_pillow_time"])


loading_time =[]
import_time =[]
import_torch_time=[]
import_torchvision_time=[]
import_pillow_time=[]

for i in range(10):
    tmp = request_importing_time(i)
    print(i, tmp)
    import_time.append((tmp[0]))
    # loading_time.append(tmp[1])
    import_torch_time.append((tmp[1]))
    import_torchvision_time.append((tmp[2]))
    import_pillow_time.append((tmp[3]))

print("import时间")
print(import_time)
print(sum(import_time) / len(import_time))

print("import-torch时间")
print(import_torch_time)
print(sum(import_torch_time) / len(import_torch_time))

print("import-torchvision时间")
print(import_torchvision_time)
print(sum(import_torchvision_time) / len(import_torchvision_time))

print("import-pillow时间")
print(import_pillow_time)
print(sum(import_pillow_time) / len(import_pillow_time))

# print("loading时间")
# print(loading_time)
# print(sum(loading_time) / len(loading_time))
