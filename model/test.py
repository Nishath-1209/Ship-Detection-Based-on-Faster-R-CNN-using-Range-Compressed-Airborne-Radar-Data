import torch
import cv2
import matplotlib.pyplot as plt
from faster_rcnn import get_model

model = get_model()
model.load_state_dict(torch.load("outputs/ship_model.pth"))
model.eval()

img = cv2.imread("data/images/boat0.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.tensor(img_rgb / 255.0).permute(2, 0, 1).float()

with torch.no_grad():
    output = model([img_tensor])

for box in output[0]["boxes"]:
    x1, y1, x2, y2 = box.int()
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.imshow(img_rgb)
plt.axis("off")
plt.show()
