import jetson.inference
import jetson.utils

import cv2
import numpy as np
import math

import argparse
import sys

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser(description="Age and gender classification in a live camera stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="",
                    nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="",
                    nargs='?', help="URI of the output stream")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

bgr_img = None
cv_img = None

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male','Female']

faceNet = cv2.dnn.readNet(faceModel,faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

padding = 20

while True:
    img = input.Capture()
    rgb_img = img
    
    if bgr_img is None:
        bgr_img = jetson.utils.cudaAllocMapped(width=rgb_img.width, height=rgb_img.height, format='bgr8')
        cv_img = jetson.utils.cudaToNumpy(bgr_img)
    
    jetson.utils.cudaConvertColor(rgb_img, bgr_img)
    jetson.utils.cudaDeviceSynchronize()

    result_img, face_boxes = highlightFace(faceNet, cv_img)
    if not face_boxes:
        print("No face detected.")
    
    for face_box in face_boxes:
        face = cv_img[max(0,face_box[1]-padding):
                   min(face_box[3]+padding,cv_img.shape[0]-1),max(0,face_box[0]-padding)
                   :min(face_box[2]+padding, cv_img.shape[1]-1)]
        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB = False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        cv2.putText(result_img, "{}, {}".format(gender, age), (face_box[0], face_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Live Steaming", result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
