import cv2
import numpy as np 


#load model
net = cv2.dnn.readNet("./model/model1/yolov3.weights", "./model/model1/yolov3.cfg")
classes = []
with open("./model/model1/label.names", "r") as f:
    classes = [line.strip() for line in f.readlines()] 
output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
#load image
img = cv2.imread("./images/kangaroo.png")
height, width, channels = img.shape
#detect objects
blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)
#box dimensions
boxes = []
confs = []
class_ids = []
for output in outputs:
    for detect in output:
        scores = detect[5:]
        class_id = np.argmax(scores)
        conf = scores[class_id]
        if conf > 0.3:
            center_x = int(detect[0] * width)
            center_y = int(detect[1] * height)
            w = int(detect[2] * width)
            h = int(detect[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confs.append(float(conf))
            class_ids.append(class_id)
#draw boxes
indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

cv2.imwrite("result4.jpg", img)

