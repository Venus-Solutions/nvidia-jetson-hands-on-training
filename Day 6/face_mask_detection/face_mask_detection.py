import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image

from network import Net

prototxt_path = "deploy.prototxt"
face_model_path = "faceDetect.caffemodel"

print("Loadding Face Model")
face_model = cv2.dnn.readNetFromCaffe(prototxt_path, face_model_path)

mask_model_path = "classifier_0.2578.pth"

print("Loading Mask Model")
mask_model = Net()
mask_model.load_state_dict(torch.load(mask_model_path))
mask_model.eval()

device = torch.device("cpu")
mask_model.to(device)

def detect_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_model.setInput(blob)
    detections = face_model.forward()
    faces=[]
    positions=[]
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX,startY)=(max(0,startX-15),max(0,startY-15))
        (endX,endY)=(min(w-1,endX+15),min(h-1,endY+15))
        confidence = detections[0, 0, i, 2]
        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            face = image[startY:endY, startX:endX]
            faces.append(face)
            positions.append((startX,startY,endX,endY))
    return faces,positions

def detect_mask(faces):
    predictions = []

    image_transforms = transforms.Compose(
        [
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if (len(faces)>0):
        for img in faces:
            img = Image.fromarray(img)
            img = image_transforms(img)
            img = img.unsqueeze(0)
            prediction = mask_model(img)
            prediction = prediction.argmax()
            predictions.append(prediction.data)
    return predictions

def main():
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        (faces, positions) = detect_face(img)
        predictions = detect_mask(faces)

        for (box, prediction) in zip(positions, predictions):
            (startX, startY, endX, endY) = box

            if prediction == 0:
                label = "Without Mask"
            else:
                label = "With Mask"

            color = (0, 255, 0) if label == "Mask" else (255,0,0)
            # color = (0, 255, 0)
            cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img,(startX, startY),(endX, endY), color, 2)

        cv2.imshow('Face Mask Detection', img)
        k = cv2.waitKey(33) # Esc
        if k == 27:
            break

if __name__ == "__main__":
    main()