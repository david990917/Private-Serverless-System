import urllib
import time

time0 = time.time()
import torch
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

import_time = time.time() - time0


def download_and_prepocess_single_try(url):
    # 保存图片预处理图片 - 研究目的
    time0 = time.time()
    filename = "target.jpg"
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    download_img_time = time.time() - time0
    print("Download_img_time: ", download_img_time)

    time0 = time.time()
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)  # 单张图片处理出来的结果

    input_batch = torch.stack([input_tensor, input_tensor], 0)  # 单张结果拼在一起

    print(input_batch.shape)
    preprocess_time = time.time() - time0
    print("Preprocess_time: ", preprocess_time)


def download_and_preprocess_single(url):
    # 下载并且处理单张图片，然后返回 tensor
    # 外面使用一个数组承接
    # 优化可以考虑多线程
    time0 = time.time()
    filename = "target.jpg"
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    download_img_time = time.time() - time0
    print("Download_img_time: ", download_img_time)

    time0 = time.time()
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)  # 单张图片处理出来的结果

    preprocess_time = time.time() - time0
    print("Preprocess_time: ", preprocess_time)
    return input_tensor

def download_and_preprocess(urlList):
    input_tensor_batch = torch.stack([download_and_preprocess_single(i) for i in urlList], 0)
    return input_tensor_batch


if __name__ == '__main__':
    # download_and_prepocess_single_try(
    #     "https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg")

    urlList = ["https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg",
               "https://dd-static.jd.com/ddimg/jfs/t1/162905/7/17846/76460/6075526dE6af30fb3/b8ed4431cb5c628a.jpg"]

    input_tensor_batch=download_and_preprocess(urlList)
    print(input_tensor_batch.shape)

    # 目前还都是串行

    # # 使用 python 的迭代器 - 0.26526427268981934
    # time0 = time.time()
    # # input_tensor0 = download_and_preprocess_single(urlList[0])
    # # input_tensor1 = download_and_preprocess_single(urlList[1])
    # input_tensor = [download_and_preprocess_single(i) for i in urlList]
    # print("total_time :", time.time() - time0)


    # # 分开计算 - 0.15804696083068848
    # time0 = time.time()
    # input_tensor0 = download_and_preprocess_single(urlList[0])
    # input_tensor1 = download_and_preprocess_single(urlList[1])
    # input_tensor = [input_tensor0, input_tensor1]
    # print("total_time :", time.time() - time0)


