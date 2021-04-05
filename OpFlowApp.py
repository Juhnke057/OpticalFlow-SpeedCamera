#Imorts
import cv2
import numpy as np

#feature parameters
feature_params = {
        'maxCorners': 100,
        'qualityLevel': 0.5,
        'minDistance': 7
}

#Camera and video intrinsics
FPS = 30
PX_PER_CM = 370

#App parameters
REFRESH_RATE = 20
DISTANCE_THRESH = 20

#Compute Euclidean distance (distance frormula)
def d2(p, q):
    return np.linalg.norm(np.array(p) - np.array(q))

#Load Video and read the first frame
video_cap = cv2.VideoCapture('test.mov')
_, frame = video_cap.read()
frame_counter = 1

#Convert first frame to grayscale and pick points to track
old_gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(image=old_gray, **feature_params)
#Show features on first frame
"""
for pt in prev_pts:
    x, y = pt.ravel()
    cv2.circle(frame, (x, y), 5, (0,255,0), -1)
cv2.imshow('features', frame)
cv2.waitKey(0)
"""

#Create a mask for the lines "Layer"
mask = np.zeros_like(frame)
#Create a mask for the speed
mask_text = np.zeros_like(frame)

#Main UI Loop
while True:
    #Reset the lines
    if frame_counter % REFRESH_RATE == 0:
        mask.fill(0)
        mask_text.fill(0)
    #Read in a video frame
    _, frame = video_cap.read()
    if frame is None:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Compute optical flow point
    next_pts, statuses, _ =cv2.calcOpticalFlowPyrLK(prevImg=old_gray, nextImg=frame_gray, prevPts=prev_pts, nextPts=None)
    
    #Keep optical flow point that are valid
    good_next_pts = next_pts[statuses == 1]
    good_old_pts = prev_pts[statuses == 1]
    
    #Draw optical flow line between old and next pt
    for good_next_pt, good_old_pt in zip(good_next_pts, good_old_pts):
        #Get new and old points
        x, y = good_next_pt.ravel()
        r, s = good_old_pt.ravel()
        
        #Draw the optical flow line
        cv2.line(mask, (x, y), (r, s), (0,0,255), 2)
        
        #Draw the coorinate of the corner points
        cv2.circle(frame, (x, y), 5, (0,0,255), -1)
        
        #Draw Speed if the distance criteria is met
        distance = d2((x, y), (r, s))
        if distance > DISTANCE_THRESH:
            #compute speed (PX * CM/PX = CM)
            speed_str = str(distance / PX_PER_CM * FPS) + ' cm/s'
            print(speed_str)
            #Puts the text on mask_text
            cv2.putText(mask_text, speed_str, (x,y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,0))
        
    #combine mask with frame
    frame_final = cv2.add(frame, mask)
    frame_final = cv2.add(frame_final, mask_text)
    
    cv2.imshow('frame', frame_final)
    
    #Update for next frame
    old_gray = frame_gray.copy()
    prev_pts = good_next_pts.reshape(-1, 1, 2)
    frame_counter += 1
    if cv2.waitKey(10) == ord('q'):
        break
    
#clean up resources
cv2.destroyAllWindows() 
video_cap.release()   
