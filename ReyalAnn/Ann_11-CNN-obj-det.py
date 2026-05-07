import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

# Load pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Read image
img = cv2.imread(r"C:\Users\Sujal\Desktop\ultimate_sem6\ReyalAnn\sample.jpg")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect objects
with torch.no_grad():
    pred = model([ToTensor()(img_rgb)])[0]

# Draw boxes
for box, score in zip(pred['boxes'], pred['scores']):
    if score > 0.5:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show result
plt.imshow(img_rgb)
plt.axis("off")
plt.show()