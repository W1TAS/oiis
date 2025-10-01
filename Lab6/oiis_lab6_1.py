import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, default='dogs.jpg',
                help='path to input image')
ap.add_argument('-c', '--config', required=False, default='yolov3.cfg',
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=False, default='yolov3.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=False, default='coco.names',
                help='path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


image = cv2.imread(args.image)

if image is None:
    print(f"Error: Could not load image '{args.image}'. Please check if the file exists.")
    exit(1)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.01
nms_threshold = 0.4

print(f"Processing {len(outs)} output layers")
total_detections = 0

for layer_idx, out in enumerate(outs):
    print(f"Layer {layer_idx}: {len(out)} detections")
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        total_detections += 1

        # Show top detections for debugging
        if confidence > 0.01:  # Show more detections
            print(
                f"  Detection: class={class_id} ({classes[class_id] if class_id < len(classes) else 'unknown'}), confidence={confidence:.4f}")

        if confidence > 0.01:  # Lower threshold
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

print(f"Total detections processed: {total_detections}")
print(f"Detections above threshold (0.01): {len(boxes)}")

# Show top 10 highest confidence detections
if len(confidences) > 0:
    sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    print("Top 10 detections by confidence:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"  {i + 1}. Class: {classes[class_ids[idx]]}, Confidence: {confidences[idx]:.4f}")
else:
    print("No detections found even with threshold 0.01")

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

print(f"Found {len(boxes)} detections")
print(f"NMS returned {len(indices) if len(indices) > 0 else 0} indices")

if len(indices) > 0:
    for i in indices:
        try:
            # Handle different return types from NMSBoxes
            if isinstance(i, (list, tuple)):
                i = i[0]
            box = boxes[i]
        except:
            continue

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        print(f"Drawing box for class {classes[class_ids[i]]} with confidence {confidences[i]:.2f}")
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
else:
    print("No objects detected after NMS")

cv2.imshow("object detection", image)
cv2.waitKey()

cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
