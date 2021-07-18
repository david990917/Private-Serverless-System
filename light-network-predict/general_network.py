# 生成通用的模型，使用model来进行调度
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
    "resnet":ResnetPath
}


def gpuResult(modelName, url):
    time0 = time.time()
    model = torch.load(networkMapping[modelName])
    model.eval()
    load_model_time = time.time() - time0
    print("Load_model_time: ", load_model_time)

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
    input_tensor = preprocess(input_image)
    print("input_tensor's shape", input_tensor.shape)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    preprocess_time = time.time() - time0
    print("Preprocess_time: ", preprocess_time)

    # # move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')

    time0 = time.time()
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    # for i in range(top5_prob.size(0)):
    #     print(categories[top5_catid[i]], top5_prob[i].item())
    inference_time = time.time() - time0
    print("Inference_time: ", inference_time)
    print(categories[top5_catid[0]], top5_prob[0].item())
    return {}


if __name__ == '__main__':
    print(import_time)

    networkName = "resnet"
    gpuResult(networkName,
              "https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg")
    print("\n\n")
    gpuResult(networkName,
              "https://dd-static.jd.com/ddimg/jfs/t1/162905/7/17846/76460/6075526dE6af30fb3/b8ed4431cb5c628a.jpg")
