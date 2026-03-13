import cv2

net = cv2.dnn.readNetFromCaffe('mobilenet/mobilenet_deploy.prototxt', 'mobilenet/mobilenet.caffemodel')
classes = []
with open('mobilenet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

image = cv2.imread('images/cat.png')
blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))

net.setInput(blob)
preds = net.forward() # ймовірність

index = preds[0].argmax()
label = classes[index] if index < len(classes) else "unknown"
conf = float(preds[0][index].item()) * 100
text = f"Class {label}: {round(conf, 2)}%"
print(text)
cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 0, 0), 2)
cv2.imshow('result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()