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

import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


# Read the categories
with open("/proxy/exec/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# print("Time for Loading pillow is : ", loading_pillow_time)

# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
# torch.save(model,PATH)


# Loading Model
time0 = time.time()
# PATH = "/proxy/exec/resnet152.pth"
# model = torch.load(PATH)
# model=resnet152()
model=resnet152()

# model = torchvision.models.resnet152()
time01 = time.time()
model.load_state_dict(torch.load("/proxy/exec/resnet152_dict.pth"))
time02 = time.time()
loading_model_time1 = time01 - time0
loading_model_time2 = time02 - time0
loading_model_time21 = time02 - time01
loading_model_totally_time = time02 - time0

# print("\n\nTime for Loading Model is :", loading_model_time)

model.eval()
loading_model_totally_time = time.time() - time0
# print("Time for Loading Model Totally is :", loading_model_totally_time)

# Download an example image from the pytorch website
import urllib

GoogleNetPATH = "/proxy/exec/googlenet.pth"
InceptionV3Path = "/proxy/exec/inceptionv3.pth"
MobileNetV2Path = "/proxy/exec/mobilenetv2.pth"
ResnetPath = "/proxy/exec/resnet152.pth"

networkMapping = {
    "googlenet": GoogleNetPATH,
    "inception": InceptionV3Path,
    "mobilenet": MobileNetV2Path,
    "resnet": ResnetPath
}


def getResult(url):
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
                "loading_model_time1": loading_model_time1,
                "loading_model_time2": loading_model_time2,
                "loading_model_time21": loading_model_time21,
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
    filename = "/proxy/exec/target.jpg"
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


def getResultList_lightmodel(modelName, url):
    # 加载模型 - load到一个新变量中
    time0 = time.time()
    light_model = torch.load(networkMapping[modelName])
    light_model.eval()
    load_light_model_time = time.time() - time0
    print("Load_light_model_time: ", load_light_model_time)

    filename = "/proxy/exec/target.jpg"
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

    return {"model": modelName,
            "result": categories[top1_catid[0]],
            "possibility": top1_prob[0].item(),
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
