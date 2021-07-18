# 可以复制给其他container的py文件
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
# print("Time for Loading pillow is : ", loading_pillow_time)

# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
# torch.save(model,PATH)

# Loading Model
time0 = time.time()
PATH = "/proxy/exec/resnet152.pth"
model = torch.load(PATH)
loading_model_time = time.time() - time0
# print("\n\nTime for Loading Model is :", loading_model_time)

model.eval()
loading_model_totally_time = time.time() - time0
# print("Time for Loading Model Totally is :", loading_model_totally_time)

# Download an example image from the pytorch website
import urllib


def gpuResult(url):
    filename = "/proxy/exec/target.jpg"

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

    # Read the categories
    with open("/proxy/exec/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

    total_loading_time = loading_torch_time + loading_torchvision_time + loading_pillow_time
    total_photo_time = downloading_time + processing_photo_time + inference_time

    return {"result": categories[top5_catid[0]],
            "possibility": top5_prob[0].item(),
            "loading_torch_time": loading_torch_time,
            "loading_torchvision_time": loading_torchvision_time,
            "loading_pillow_time": loading_pillow_time,
            "loading_model_time": loading_model_totally_time,
            "downloading_photo_time": downloading_time,
            "processing_photo_time": processing_photo_time,
            "inference_time": inference_time,
            "total_loading_time": total_loading_time,
            "total_photo_time": total_photo_time,
            "total_time": total_photo_time + total_loading_time
            }


if __name__ == '__main__':
    result = gpuResult(
        "https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg")
    print(result)
