import math
from collections import OrderedDict
#from nets.darknet import darknet53
import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

#https://github.com/bubbliiiing/yolo3-pytorch
#yolov3模型参考代码
#https://cloud.tencent.com/developer/article/2132010
#代码模型构建部分解析，yolov3模型讲解
#https://www.bilibili.com/video/BV1Hp4y1y788/?spm_id_from=333.337.search-card.all.click&vd_source=487e709b64443fab2276b1551be4170f
#Pytorch 搭建自己的YOLO3目标检测平台（Bubbliiiing 深度学习 教程）--b站yolov3模型的介绍以及如何实现，可参考

# ---------------------------------------------------------------------#
# 残差结构
# 利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
# 最后接上一个残差边
# ---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # ---------------------------------------------------------------------#
    # 在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    # 然后进行残差结构的堆叠
    # ---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model


def conv2d(filter_in, filter_out, kernel_size):
 pad = (kernel_size - 1) // 2 if kernel_size else 0
 return nn.Sequential(OrderedDict([
  ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
  ("bn", nn.BatchNorm2d(filter_out)),
  ("relu", nn.LeakyReLU(0.1)),
 ]))


# ------------------------------------------------------------------------#
# make_last_layers里面一共有七个卷积，前五个用于提取特征。
# 后两个用于获得yolo网络的预测结果
# ------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
 m = nn.Sequential(
  conv2d(in_filters, filters_list[0], 1),
  conv2d(filters_list[0], filters_list[1], 3),
  conv2d(filters_list[1], filters_list[0], 1),
  conv2d(filters_list[0], filters_list[1], 3),
  conv2d(filters_list[1], filters_list[0], 1),
  conv2d(filters_list[0], filters_list[1], 3),
  nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
 )
 return m


class YoloBody(nn.Module):
 def __init__(self, anchors_mask, num_classes):
  super(YoloBody, self).__init__()
  # ---------------------------------------------------#
  # 生成darknet53的主干模型
  # 获得三个有效特征层，他们的shape分别是：
  # 52,52,256
  # 26,26,512
  # 13,13,1024
  # ---------------------------------------------------#
  self.backbone = darknet53()

  # ---------------------------------------------------#
  # out_filters : [64, 128, 256, 512, 1024]
  # ---------------------------------------------------#
  out_filters = self.backbone.layers_out_filters

  # ------------------------------------------------------------------------#
  # 计算yolo_head的输出通道数，对于voc数据集而言
  # final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
  # ------------------------------------------------------------------------#
  self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

  self.last_layer1_conv = conv2d(512, 256, 1)
  self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
  self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

  self.last_layer2_conv = conv2d(256, 128, 1)
  self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
  self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

 def forward(self, x):
  # ---------------------------------------------------#
  # 获得三个有效特征层，他们的shape分别是：
  # 52,52,256；26,26,512；13,13,1024
  # ---------------------------------------------------#
  x2, x1, x0 = self.backbone(x)

  # ---------------------------------------------------#
  # 第一个特征层
  # out0 = (batch_size,255,13,13)
  # ---------------------------------------------------#
  # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
  out0_branch = self.last_layer0[:5](x0)
  out0 = self.last_layer0[5:](out0_branch)

  # 13,13,512 -> 13,13,256 -> 26,26,256
  x1_in = self.last_layer1_conv(out0_branch)
  x1_in = self.last_layer1_upsample(x1_in)

  # 26,26,256 + 26,26,512 -> 26,26,768
  x1_in = torch.cat([x1_in, x1], 1)
  # ---------------------------------------------------#
  # 第二个特征层
  # out1 = (batch_size,255,26,26)
  # ---------------------------------------------------#
  # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
  out1_branch = self.last_layer1[:5](x1_in)
  out1 = self.last_layer1[5:](out1_branch)

  # 26,26,256 -> 26,26,128 -> 52,52,128
  x2_in = self.last_layer2_conv(out1_branch)
  x2_in = self.last_layer2_upsample(x2_in)

  # 52,52,128 + 52,52,256 -> 52,52,384
  x2_in = torch.cat([x2_in, x2], 1)
  # ---------------------------------------------------#
  # 第一个特征层
  # out3 = (batch_size,255,52,52)
  # ---------------------------------------------------#
  # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
  out2 = self.last_layer2(x2_in)
  return out0, out1, out2





class DecodeBox():
 def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
  super(DecodeBox, self).__init__()
  self.anchors = anchors
  self.num_classes = num_classes
  self.bbox_attrs = 5 + num_classes
  self.input_shape = input_shape
  # -----------------------------------------------------------#
  # 13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
  # 26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
  # 52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
  # -----------------------------------------------------------#
  self.anchors_mask = anchors_mask

 def decode_box(self, inputs):
  outputs = []
  for i, input in enumerate(inputs):
   # -----------------------------------------------#
   # 输入的input一共有三个，他们的shape分别是
   # batch_size, 255, 13, 13
   # batch_size, 255, 26, 26
   # batch_size, 255, 52, 52
   # -----------------------------------------------#
   batch_size = input.size(0)
   input_height = input.size(2)
   input_width = input.size(3)

   # -----------------------------------------------#
   # 输入为416x416时
   # stride_h = stride_w = 32、16、8
   # -----------------------------------------------#
   stride_h = self.input_shape[0] / input_height
   stride_w = self.input_shape[1] / input_width
   # -------------------------------------------------#
   # 此时获得的scaled_anchors大小是相对于特征层的
   # -------------------------------------------------#
   scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                     self.anchors[self.anchors_mask[i]]]

   # -----------------------------------------------#
   # 输入的input一共有三个，他们的shape分别是
   # batch_size, 3, 13, 13, 85
   # batch_size, 3, 26, 26, 85
   # batch_size, 3, 52, 52, 85
   # -----------------------------------------------#
   prediction = input.view(batch_size, len(self.anchors_mask[i]),
                           self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

   # -----------------------------------------------#
   # 先验框的中心位置的调整参数
   # -----------------------------------------------#
   x = torch.sigmoid(prediction[..., 0])
   y = torch.sigmoid(prediction[..., 1])
   # -----------------------------------------------#
   # 先验框的宽高调整参数
   # -----------------------------------------------#
   w = prediction[..., 2]
   h = prediction[..., 3]
   # -----------------------------------------------#
   # 获得置信度，是否有物体
   # -----------------------------------------------#
   conf = torch.sigmoid(prediction[..., 4])
   # -----------------------------------------------#
   # 种类置信度
   # -----------------------------------------------#
   pred_cls = torch.sigmoid(prediction[..., 5:])

   FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
   LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

   # ----------------------------------------------------------#
   # 生成网格，先验框中心，网格左上角
   # batch_size,3,13,13
   # ----------------------------------------------------------#
   grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
    batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
   grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
    batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

   # ----------------------------------------------------------#
   # 按照网格格式生成先验框的宽高
   # batch_size,3,13,13
   # ----------------------------------------------------------#
   anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
   anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
   anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
   anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

   # ----------------------------------------------------------#
   # 利用预测结果对先验框进行调整
   # 首先调整先验框的中心，从先验框中心向右下角偏移
   # 再调整先验框的宽高。
   # ----------------------------------------------------------#
   pred_boxes = FloatTensor(prediction[..., :4].shape)
   pred_boxes[..., 0] = x.data + grid_x
   pred_boxes[..., 1] = y.data + grid_y
   pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
   pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

   # ----------------------------------------------------------#
   # 将输出结果归一化成小数的形式
   # ----------------------------------------------------------#
   _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
   output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                       conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
   outputs.append(output.data)
  return outputs

 def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
  # -----------------------------------------------------------------#
  # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
  # -----------------------------------------------------------------#
  box_yx = box_xy[..., ::-1]
  box_hw = box_wh[..., ::-1]
  input_shape = np.array(input_shape)
  image_shape = np.array(image_shape)

  if letterbox_image:
   # -----------------------------------------------------------------#
   # 这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
   # new_shape指的是宽高缩放情况
   # -----------------------------------------------------------------#
   new_shape = np.round(image_shape * np.min(input_shape / image_shape))
   offset = (input_shape - new_shape) / 2. / input_shape
   scale = input_shape / new_shape

   box_yx = (box_yx - offset) * scale
   box_hw *= scale

  box_mins = box_yx - (box_hw / 2.)
  box_maxes = box_yx + (box_hw / 2.)
  boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
  boxes *= np.concatenate([image_shape, image_shape], axis=-1)
  return boxes

 def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                         nms_thres=0.4):
  # ----------------------------------------------------------#
  # 将预测结果的格式转换成左上角右下角的格式。
  # prediction [batch_size, num_anchors, 85]
  # ----------------------------------------------------------#
  box_corner = prediction.new(prediction.shape)
  box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
  box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
  box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
  box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
  prediction[:, :, :4] = box_corner[:, :, :4]

  output = [None for _ in range(len(prediction))]
  for i, image_pred in enumerate(prediction):
   # ----------------------------------------------------------#
   # 对种类预测部分取max。
   # class_conf [num_anchors, 1] 种类置信度
   # class_pred [num_anchors, 1] 种类
   # ----------------------------------------------------------#
   class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

   # ----------------------------------------------------------#
   # 利用置信度进行第一轮筛选
   # ----------------------------------------------------------#
   conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

   # ----------------------------------------------------------#
   # 根据置信度进行预测结果的筛选
   # ----------------------------------------------------------#
   image_pred = image_pred[conf_mask]
   class_conf = class_conf[conf_mask]
   class_pred = class_pred[conf_mask]
   if not image_pred.size(0):
    continue
   # -------------------------------------------------------------------------#
   # detections [num_anchors, 7]
   # 7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
   # -------------------------------------------------------------------------#
   detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

   # ------------------------------------------#
   # 获得预测结果中包含的所有种类
   # ------------------------------------------#
   unique_labels = detections[:, -1].cpu().unique()

   if prediction.is_cuda:
    unique_labels = unique_labels.cuda()
    detections = detections.cuda()

   for c in unique_labels:
    # ------------------------------------------#
    # 获得某一类得分筛选后全部的预测结果
    # ------------------------------------------#
    detections_class = detections[detections[:, -1] == c]

    # ------------------------------------------#
    # 使用官方自带的非极大抑制会速度更快一些！
    # ------------------------------------------#
    keep = nms(
     detections_class[:, :4],
     detections_class[:, 4] * detections_class[:, 5],
     nms_thres
    )
    max_detections = detections_class[keep]

    # # 按照存在物体的置信度排序
    # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
    # detections_class = detections_class[conf_sort_index]
    # # 进行非极大抑制
    # max_detections = []
    # while detections_class.size(0):
    # # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
    # max_detections.append(detections_class[0].unsqueeze(0))
    # if len(detections_class) == 1:
    # break
    # ious = bbox_iou(max_detections[-1], detections_class[1:])
    # detections_class = detections_class[1:][ious < nms_thres]
    # # 堆叠
    # max_detections = torch.cat(max_detections).data

    # Add max detections to outputs
    output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

   if output[i] is not None:
    output[i] = output[i].cpu().numpy()
    box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
    output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
  return output

 if __name__ == "__main__":
     anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
     model = YoloBody(anchors_mask, 2)
     print(model)

