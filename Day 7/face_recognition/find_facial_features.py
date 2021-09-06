from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file("Biden_Obama.jpg")

face_landmarks_list = face_recognition.face_landmarks(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:

    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5)

pil_image.show()