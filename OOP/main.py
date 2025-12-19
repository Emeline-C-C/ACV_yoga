from engine import Engine 
import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
import keras

#open window
engine = Engine()
compteur = 0
#open loop: get frames
while True:
    compteur += 1


    ret, frame = engine.get_image() 
    engine.frame = frame
    if not ret:
        print("Failed to grab frame")
        break

    
    engine.print_image() #cv2.imshow(self.window_name, frame)
    
    if compteur%5==0:

        engine.get_landmarks(frame)
        
        input_vector = engine.flatten_landmarks()

        if input_vector.shape == (132, 1):

            engine.model(input_vector)
            engine.print_pose()

    engine.print_image() 
  

        
        
 
    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

engine.release_webcam() 
cv2.destroyAllWindows()
#get landmarks from frame
#End loop: predict poses
#Terminate games


