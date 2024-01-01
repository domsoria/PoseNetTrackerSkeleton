import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO

# Define the skeleton structure
SKELETON_EDGES = [
    # Definisci le connessioni tra i keypoints qui
    # Esempio: (0, 1), (1, 2), ...
    #(0, 1), (0, 2), (2, 4), (1, 3), (5, 6), (6, 8), (8,10), (5, 7), (7,9), (6,12), (5,11), (11,12), (11,13), (13,15), (12,14), (14,16)
    (0, 16), (1,16), (0, 2), (1, 3), (4, 5), (4,6),(6, 8), (5, 7), (7,9), (4,10),(10,12), (5,11), (11,10), (11,13), (13,15), (12,14)
]

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
print("scegli tra video o webcam")
print("1: video")
print("2: webcam")
scelta = input()
if scelta == "1":
    #print("inserisci il path del video")
    #path = input()
    path = "./test_video.mp4"
    cap = cv2.VideoCapture(path)
elif scelta == "2":
    cap = cv2.VideoCapture(0)
    set_res = cap.set(3, 1280), cap.set(4, 720) # set resolution (1280x720, 640x480, 320x200

save_path = "./test_video.txt"
f = open(save_path, 'a') # Modalità 'append'
        
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Predict with the model
    #model.predict(source=source, show=True, conf=0.3, save=True)
    #results = model.predict(frame,show=False, conf=0.3, save=False)  # predict on a frame
    results = model.track(frame, verbose=False,  show=False, persist=True, conf=0)
    
    # For each person detected in the frame
    if True:
        for result in results:
            kpts = result.keypoints
            boxes = result.boxes
            masks = result.masks
            nk = kpts.shape[1]

            if(boxes.id is not None):
                kpt_insts = kpts.data
                box_insts = boxes.xyxy[:, :4]
                confs_insts = boxes.data[:, 4]
                #print(box_insts.tolist())
                #print(boxes.id.tolist())
                for box in box_insts.tolist():
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                x = [0] * 17
                y = [0] * 17
                for kpt_inst in kpt_insts.tolist():
                    #calola l'indice del kpt_inst nel for
                    for indice in range(17):
                        x[indice], y[indice] = (int(kpt_inst[-16+indice][0]), int(kpt_inst[-16 + indice][1]))   
                        cv2.circle(frame, (x[indice], y[indice]), 5, (0, 255, 0), -1)
                        cv2.putText(frame, str(indice), (x[indice], y[indice]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    # Draw the skeleton
                    for edge in SKELETON_EDGES:
                        # Skip edges where either of the keypoints is not detected 
                        if edge[0] >= nk or edge[1] >= nk:
                            continue
                        if (x[edge[0]], y[edge[0]]) != (0, 0) and (x[edge[1]], y[edge[1]]) != (0, 0):
                             cv2.line(frame, (x[edge[0]], y[edge[0]]),(x[edge[1]],y[edge[1]]),(255, 0, 0), 2)      
           
        # Display the resulting frame
        cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
