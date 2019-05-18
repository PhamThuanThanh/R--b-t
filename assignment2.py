import cv2 as cv
import numpy as np
import click
import time 

import ai2thor.controller

controller = ai2thor.controller.Controller()
controller.start(player_screen_width=500, player_screen_height=400)
controller.reset('FloorPlan30')
event = controller.step(dict(action='Initialize', gridSize=0.25, rotation=0, horizon=0))

moves = {'a': 'MoveLeft', 'w': 'MoveAhead', 'd': 'MoveRight', 's': 'MoveBack'}
rotates = {'j': -10, 'l': 10}
looks = {'i': -10, 'k': 10}

rotate = 0
horizon = 0

winName = 'Detect Object'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 500,500)

#Initialize the parameters
confThreshold = 0.25
nmsThreshold = 0.40
inpWidth = 416
inpHeight = 416
	
#load names of classes
classesFile = "yolov3-thor.names"
classes = None
with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#load the network file cfg and weight
modelConf = 'yolov3-thor.cfg'
modelWeights = 'yolov3-thor.weights'

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

#remove bouding boxes
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)
                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight)
                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    
    #non maximum suppression
    indices = cv.dnn.NMSBoxes(boxes,confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]      
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)

#Draw predicted bouding boxes
def drawPred(classId, conf, left, top, right, bottom):

    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

    label = '%.2f' % conf

    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.putText(frame, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#get names of output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

while True:

    frame = cv.resize(event.cv2img, (400, 400))

    #create blob from input image
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

    #Set the input the the net
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)

    #show overall time for inference(t) and each of layers
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv.imshow(winName, frame)

    key = cv.waitKey(0)
    key = chr(key)
    
    #move agent
    if key == 'q':
        break
    elif key in moves:
        event = controller.step(dict(action=moves[key]))
    elif key in rotates:
        rotate += rotates[key]
        event = controller.step(action=dict(action='Rotate', rotation=rotate))
    elif key in looks:
        horizon += looks[key]
        event = controller.step(action=dict(action='Look', horizon=horizon))

cv.destroyAllWindows()



