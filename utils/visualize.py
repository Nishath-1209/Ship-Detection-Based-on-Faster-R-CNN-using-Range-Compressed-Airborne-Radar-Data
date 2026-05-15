import cv2
import xml.etree.ElementTree as ET

img = cv2.imread("data/images/boat1.png")
tree = ET.parse("data/annotations/boat1.xml")
root = tree.getroot()

for obj in root.findall("object"):
    b = obj.find("bndbox")
    x1 = int(b.find("xmin").text)
    y1 = int(b.find("ymin").text)
    x2 = int(b.find("xmax").text)
    y2 = int(b.find("ymax").text)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow("Ship Annotation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
