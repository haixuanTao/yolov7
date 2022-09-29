import os

import cv2
import numpy as np
import torch

from yolov7_tt100k import (
    WEIGHTS,
    letterbox,
    non_max_suppression,
    plot_one_box,
    scale_coords,
)

model = torch.load(WEIGHTS)  # load
model = model["ema" if model.get("ema") else "model"].float().fuse().eval()
image = os.path.join(os.path.dirname(__file__), "2.jpg")

img = cv2.imread(image)
im0 = cv2.imread(image)
img, ratio, (dw, dh) = letterbox(img)
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to(torch.device("cuda"))
img = img.half() if False else img.float()  # uint8 to fp16/32
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

pred = model(img)[0]
pred = non_max_suppression(pred, 0.15, 0.15, None, False)[0]

pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
for *xyxy, conf, cls in reversed(pred):
    plot_one_box(xyxy, im0, label="res", color=int(cls), line_thickness=1)
print(pred.detach().cpu().numpy())

cv2.imshow(str(1), im0)
cv2.waitKey(10000)  # 1 millisecond
