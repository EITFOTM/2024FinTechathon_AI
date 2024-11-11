# YOLO 检测网络后处理技术文档

## 类概述
`DecodeBox` 类实现了 YOLO 目标检测网络的后处理操作，主要包含以下功能：
- 检测框解码
- 检测框坐标修正
- 非极大值抑制(NMS)

## 1. 初始化阶段

```python
def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6,7,8], [3,4,5], [0,1,2]]):
    self.anchors = anchors          # 预设的锚框
    self.num_classes = num_classes  # 类别数量
    self.bbox_attrs = 5 + num_classes  # 边界框属性数(x,y,w,h,conf + 类别数)
    self.input_shape = input_shape  # 输入图像尺寸
    self.anchors_mask = anchors_mask  # 不同特征层使用的锚框索引
```

初始化阶段设置了以下重要参数：
- anchors：预设的锚框尺寸
- anchors_mask：三个特征层(13x13、26x26、52x52)分别使用的锚框索引
- input_shape：网络输入图像的尺寸
- bbox_attrs：每个边界框的属性数量(4个坐标 + 1个置信度 + 类别数)

## 2. 检测框解码阶段

核心方法 `decode_box` 负责将网络输出转换为实际的检测框坐标：

```python
def decode_box(self, inputs):
    # 处理三个尺度的特征图输出
    for i, input in enumerate(inputs):
        # 计算特征图的步长
        stride_h = self.input_shape[0] / input_height
        stride_w = self.input_shape[1] / input_width
        
        # 将锚框缩放到特征图尺度
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) 
                         for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]
```

关键步骤：
1. 重塑网络输出维度
2. 提取边界框参数(x,y,w,h)和置信度
3. 生成特征图网格点坐标
4. 计算实际检测框的位置和大小

### 检测框坐标计算
```python
# 预测框中心点偏移
pred_boxes[..., 0] = x.data + grid_x
pred_boxes[..., 1] = y.data + grid_y
# 预测框宽高计算
pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
```

## 3. 坐标修正阶段

`yolo_correct_boxes` 方法处理图像预处理带来的尺寸变化：

```python
def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # 处理letterbox导致的坐标偏移
    if letterbox_image:
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
```

主要功能：
- 计算图像缩放比例
- 修正letterbox填充带来的偏移
- 将坐标转换回原始图像尺度

## 4. 非极大值抑制阶段

`non_max_suppression` 方法用于去除冗余的检测框：

```python
def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, 
                       letterbox_image, conf_thres=0.5, nms_thres=0.4):
    # 转换预测框格式为左上角右下角
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
```

关键步骤：
1. 将预测框格式从中心点+宽高转换为左上角+右下角
2. 按置信度阈值筛选检测框
3. 对每个类别分别进行NMS处理
4. 将保留的检测框转换回原始图像坐标系

### NMS处理流程
```python
# 获取每个类别的检测结果
detections_class = detections[detections[:, -1] == c]

# 执行NMS
keep = nms(
    detections_class[:, :4],
    detections_class[:, 4] * detections_class[:, 5],
    nms_thres
)
max_detections = detections_class[keep]
```

## 参数说明

### 输入参数
- conf_thres：置信度阈值，默认0.5
- nms_thres：NMS阈值，默认0.4
- letterbox_image：是否使用letterbox缩放，布尔值

### 输出格式
最终输出的检测结果包含以下信息：
- boxes：检测框坐标 [x1, y1, x2, y2]
- scores：检测框置信度
- classes：检测框类别

## 使用示例

```python
if __name__ == "__main__":
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model = YoloBody(anchors_mask, 2)
    # 创建解码器实例
    decoder = DecodeBox(anchors, num_classes, input_shape, anchors_mask)
    # 执行检测框解码
    outputs = decoder.decode_box(predictions)
    # 执行NMS
    results = decoder.non_max_suppression(outputs, ...)
```
