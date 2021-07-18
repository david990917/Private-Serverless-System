import time

# Loading torch
time0 = time.time()
import torch

loading_torch_time = time.time() - time0
# print("Time for Loading torch is : ", loading_torch_time)

# Loading torchvision
time0 = time.time()
from torchvision import transforms

loading_torchvision_time = time.time() - time0
# print("Time for Loading torchvision is : ", loading_torchvision_time)

# Loading pillow
time0 = time.time()
from PIL import Image

loading_pillow_time = time.time() - time0

# Read the categories
with open("exec/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# print("Time for Loading pillow is : ", loading_pillow_time)

# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
# torch.save(model,PATH)

# Loading Model
time0 = time.time()
PATH = "exec/resnet152.pth"
model = torch.load(PATH)
loading_model_time = time.time() - time0
# print("\n\nTime for Loading Model is :", loading_model_time)

model.eval()
loading_model_totally_time = time.time() - time0
# print("Time for Loading Model Totally is :", loading_model_totally_time)

# Download an example image from the pytorch website
import urllib

GoogleNetPATH = "exec/googlenet.pth"
InceptionV3Path = "exec/inceptionv3.pth"
MobileNetV2Path = "exec/mobilenetv2.pth"
ResnetPath = "exec/resnet152.pth"

networkMapping = {
    "googlenet": GoogleNetPATH,
    "inception": InceptionV3Path,
    "mobilenet": MobileNetV2Path,
    "resnet": ResnetPath
}


def getResult(url):
    filename = "target.jpg"

    # Retreving Photos
    time0 = time.time()
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    downloading_time = time.time() - time0
    # print("\n\nTime for Downloading Target Photo is :", downloading_time)

    # Processing Photos
    time0 = time.time()
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    processing_photo_time = time.time() - time0
    # print("\n\nTime for processing Photo is :", processing_photo_time)
    # print("input_tensor's shape", input_tensor.shape)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # # move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')

    # Inference
    time0 = time.time()
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
    inference_time = time.time() - time0
    # print("\n\nTime for Inference is :", inference_time, "\n\n")

    # Show top categories per image
    # top5_prob, top5_catid = torch.topk(probabilities, 5)
    # for i in range(top5_prob.size(0)):
    #     print(categories[top5_catid[i]], top5_prob[i].item())
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    # print(top1_prob)
    # print(categories[top1_catid], top1_prob.item())

    total_import_time = loading_torch_time + loading_torchvision_time + loading_pillow_time
    total_photo_time = downloading_time + processing_photo_time + inference_time

    return {"result": categories[top1_catid[0]],
            "possibility": top1_prob[0].item(),
            "time": {
                "loading_model_time": loading_model_totally_time,
                "import_time": {
                    "loading_torch_time": loading_torch_time,
                    "loading_torchvision_time": loading_torchvision_time,
                    "loading_pillow_time": loading_pillow_time,
                    "total_import_time": total_import_time,
                },
                "photo_time": {
                    "downloading_photo_time": downloading_time,
                    "processing_photo_time": processing_photo_time,
                    "inference_time": inference_time,
                },
            },
            "total_photo_time": total_photo_time,
            "total_time(import+photo)": total_photo_time + total_import_time,
            }


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


def getResultList(urlList):
    # 加载模型 - 都要重新 load 模型吗？
    # 要不不支持了，取消重新 load 模型。等终极版本的时候再来
    # time0 = time.time()
    # model = torch.load(networkMapping[modelName])
    # model.eval()
    # load_model_time = time.time() - time0
    # print("Load_model_time: ", load_model_time)

    time0 = time.time()
    input_tensor_batch = download_and_preprocess(urlList)
    download_and_preprocess_time = time.time() - time0
    print("Download_and_Preprocess_time: ", download_and_preprocess_time)

    time0 = time.time()
    with torch.no_grad():
        output = model(input_tensor_batch)  # [batch_size,1000]

    # 修改传进去的输入为 output，维度改为1，然后得到得到 [batch_size,1000]
    # 现在的结果是正确的啦
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # 输出的结果就是 [batch_size,1]
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    result_acc = []
    for i in range(len(top1_prob)):
        result_acc.append((categories[top1_catid[i]], top1_prob[i].item()))
        print(categories[top1_catid[i]], top1_prob[i].item())

    inference_time = time.time() - time0
    print("Inference_time: ", inference_time)
    total_import_time = loading_torch_time + loading_torchvision_time + loading_pillow_time

    return {"result": result_acc,
            "time": {
                "loading_model_time": loading_model_totally_time,
                "import_time": {
                    "loading_torch_time": loading_torch_time,
                    "loading_torchvision_time": loading_torchvision_time,
                    "loading_pillow_time": loading_pillow_time,
                    "total_import_time": total_import_time,
                },
                "photo_time": {
                    "all_download_and_preprocess_time": download_and_preprocess_time,
                    "all_inference_time": inference_time,
                },
            },

            }


def getResultList_lightmodel(modelName,url):
    # 加载模型 - load到一个新变量中
    time0 = time.time()
    light_model = torch.load(networkMapping[modelName])
    light_model.eval()
    load_light_model_time = time.time() - time0
    print("Load_light_model_time: ", load_light_model_time)


    filename="target.jpg"
    # Retreving Photos
    time0 = time.time()
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    downloading_time_lightmodel = time.time() - time0

    # Processing Photos
    time0 = time.time()
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    processing_photo_time_lightmodel = time.time() - time0
    download_and_preprocess_time_lightmodel = time.time() - time0



    with torch.no_grad():
        time0 = time.time()
        output = model(input_batch)
        inference_time_lightmodel = time.time() - time0

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 输出的结果就是 [batch_size,1]
    top1_prob, top1_catid = torch.topk(probabilities, 1)


    total_import_time = loading_torch_time + loading_torchvision_time + loading_pillow_time


    return {"model":modelName,
            "result": categories[top1_catid[0]],
            "possibility":top1_prob[0].item(),
            "time": {
                "loading_model_time": load_light_model_time,
                "import_time": {
                    "loading_torch_time": loading_torch_time,
                    "loading_torchvision_time": loading_torchvision_time,
                    "loading_pillow_time": loading_pillow_time,
                    "total_import_time": total_import_time,
                },
                "photo_time": {
                    "all_download_and_preprocess_time": download_and_preprocess_time_lightmodel,
                    "all_inference_time": inference_time_lightmodel,
                },
            },

            }



if __name__ == '__main__':
    getResult("https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg")
