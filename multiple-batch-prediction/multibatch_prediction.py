# 批处理 batch - prediction
# 代码继承自 /light-network-predict/general_network
# modelPath 使用的是 light-network-predict 文件夹
import urllib
import time

time0 = time.time()
import torch
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

import_time = time.time() - time0

GoogleNetPATH = "/Users/starky/PycharmProjects/docker-torch/light-network-predict/googlenet.pth"
InceptionV3Path = "/Users/starky/PycharmProjects/docker-torch/light-network-predict/inceptionv3.pth"
MobileNetV2Path = "/Users/starky/PycharmProjects/docker-torch/light-network-predict/mobilenetv2.pth"
ResnetPath = "/Users/starky/PycharmProjects/docker-torch/light-network-predict/resnet152.pth"

networkMapping = {
    "googlenet": GoogleNetPATH,
    "inception": InceptionV3Path,
    "mobilenet": MobileNetV2Path,
    "resnet": ResnetPath
}

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


def download_and_preprocess_single(url):
    # 下载并且处理单张图片, 然后返回 tensor , 外面使用一个数组承接
    # 搬运过来的时候删除了 print
    filename = "target.jpg"
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)  # 单张图片处理出来的结果

    return input_tensor


def download_and_preprocess(urlList):
    input_tensor_batch = torch.stack([download_and_preprocess_single(i) for i in urlList], 0)
    return input_tensor_batch


def gpuResult(modelName, urlList):
    # 加载模型 - 不用修改
    time0 = time.time()
    model = torch.load(networkMapping[modelName])
    model.eval()
    load_model_time = time.time() - time0
    print("Load_model_time: ", load_model_time)

    time0 = time.time()
    input_tensor_batch = download_and_preprocess(urlList)
    download_and_preprocess_time = time.time() - time0
    print("Download_and_Preprocess_time: ", download_and_preprocess_time)

    time0 = time.time()
    with torch.no_grad():
        output = model(input_tensor_batch)  # [batch_size,1000]
    # print(output)
    # print(output[0])

    # 修改传进去的输入为 output，维度改为1，然后得到得到 [batch_size,1000]
    # 现在的结果是正确的啦
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # 输出的结果就是 [batch_size,1]
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    # print(top1_prob)
    # print(top1_catid)
    for i in range(len(top1_prob)):
        print(categories[top1_catid[i]], top1_prob[i].item())

    inference_time = time.time() - time0
    print("Inference_time: ", inference_time)

    return {}


if __name__ == '__main__':
    print(import_time)

    networkName = "resnet"
    # networkName = "mobilenet"
    urlList = ["https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg",
               "https://dd-static.jd.com/ddimg/jfs/t1/162905/7/17846/76460/6075526dE6af30fb3/b8ed4431cb5c628a.jpg"]
    url = "https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg"

    gpuResult(networkName, urlList*6)
