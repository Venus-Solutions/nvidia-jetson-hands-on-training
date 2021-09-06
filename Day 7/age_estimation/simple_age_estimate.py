import cv2

age_buckets = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
age_detect = cv2.dnn.readNet('deploy_age.prototxt', 'age_net.caffemodel')

image = cv2.imread('Joe_Biden.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_detect.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces:
	face = image[y - 10: y + h + 10, x - 10: x + w + 10][:, :, ::-1]
	
faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB = False)
age_detect.setInput(faceBlob)
prediction = age_detect.forward()

print(f'Age predicted: {age_buckets[prediction[0].argmax()]}')
print(f'Confidence %: {prediction[0][prediction[0].argmax()] * 100}')
