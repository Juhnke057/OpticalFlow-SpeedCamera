#Imort
import cv2
import numpy as np

#parametrar
feature_params = {
        'maxCorners': 100,
        'qualityLevel': 0.5,
        'minDistance': 7
}

#Kamera/Video specs.
FPS = 30
PX_PER_CM = 370

#Applikations parametrar
REFRESH_RATE = 20
DISTANCE_THRESH = 20

#räkna Euclidean distance (avstånds formel)
def d2(p, q):
    return np.linalg.norm(np.array(p) - np.array(q))

#ladda video och första frame
video_cap = cv2.VideoCapture('test.mov')
_, frame = video_cap.read()
frame_counter = 1

#Convertera frame till grayscale och välj points att följa
old_gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(image=old_gray, **feature_params)

"""
for pt in prev_pts:
    x, y = pt.ravel()
    cv2.circle(frame, (x, y), 5, (0,255,0), -1)
cv2.imshow('features', frame)
cv2.waitKey(0)
"""

#Mask för att rita linjer
mask = np.zeros_like(frame)
#Create a mask for the speed
mask_text = np.zeros_like(frame)

#UI loop
while True:
    #Reset för linjer
    if frame_counter % REFRESH_RATE == 0:
        mask.fill(0)
        mask_text.fill(0)
    #läsa in video frame
    _, frame = video_cap.read()
    if frame is None:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Compute optical flow point
    next_pts, statuses, _ =cv2.calcOpticalFlowPyrLK(prevImg=old_gray, nextImg=frame_gray, prevPts=prev_pts, nextPts=None)
    
    #Behåll godtyckliga punkter
    good_next_pts = next_pts[statuses == 1]
    good_old_pts = prev_pts[statuses == 1]
    
    #Rita linje mellan gammal point och ny
    for good_next_pt, good_old_pt in zip(good_next_pts, good_old_pts):
        #Get new and old points
        x, y = good_next_pt.ravel()
        r, s = good_old_pt.ravel()
        
        #Rita linjerna
        cv2.line(mask, (x, y), (r, s), (0,0,255), 2)
        
        #Rita kordinater
        cv2.circle(frame, (x, y), 5, (0,0,255), -1)
        
        #Sätt ut hastihet om godtyckligt
        distance = d2((x, y), (r, s))
        if distance > DISTANCE_THRESH:
            #(PX * CM/PX = CM)
            speed_str = str(distance / PX_PER_CM * FPS) + ' cm/s'
            print(speed_str)
            #Puts the text on mask_text
            cv2.putText(mask_text, speed_str, (x,y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,0))
        
    #Lägg masken över videon
    frame_final = cv2.add(frame, mask)
    frame_final = cv2.add(frame_final, mask_text)
    
    cv2.imshow('frame', frame_final)
    
    #Uppdatera nästa frame
    old_gray = frame_gray.copy()
    prev_pts = good_next_pts.reshape(-1, 1, 2)
    frame_counter += 1
    if cv2.waitKey(10) == ord('q'):
        break
    
#Rensa
cv2.destroyAllWindows() 
video_cap.release()   
