## Libraries 

import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
import keras



class Engine:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.window_name = "Yoga postures"
        self.window = cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.frame = None # attribute will be init smewhere in the main loop.
        self.pose_name = ''
      
    def get_image(self):
        self.cap = self.camera
        return self.cap.read()
    

    def print_image(self):
        cv2.imshow(self.window_name, self.frame)

        """
    Draw a rectangle with the detected pose name on the frame.
    """
         # Draw filled rectangle (background)
         # Rectangle position
        x, y, w, h = 20, 20, 320, 60
        cv2.rectangle(
            self.frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),   # green background
            -1           # filled
        )

        # Draw rectangle border
        cv2.rectangle(
            self.frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 0), # black border
            2
        )
    
          # Black pose name text
        cv2.putText(
            self.frame,
            self.pose_name,
            (x + 10, y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),  # green text
            2,
            cv2.LINE_AA
        )
        

       
        

        

      

        
    def get_landmarks(self,frame):
        """
        Extract pose landmarks as a NumPy array:
        Each landmark: [x_pixel, y_pixel, z, visibility]
        """
        landmarks_list = []
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
        

        # Change BGR to RGB (OpenCv gives you BGR and you need RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #we process the frame with MediaPipe
        results = self.pose.process(rgb_frame)

        #get landmarks from rgb_frame
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                 landmarks_list.append((lm.x, lm.y, lm.z))
        self.landmarks = np.array(landmarks_list)
    
    

    def flatten_landmarks(self):
        """
    Convert landmarks (33,4) into a 1D vector (132,)
    Returns zeros if no landmarks detected.
    """
        if self.landmarks is not None:
            input_vector = self.landmarks.flatten().reshape(1, -1)
        else:
            input_vector = np.zeros(33*4).reshape(1, -1)

        return input_vector
    
    def model(self,input_vector):
        model = tf.keras.models.load_model('./ACV_yoga/yoga_pose_model.keras')
        model.weights

        prediction = model.predict(input_vector, verbose=0)
        self.predicted_idx = np.argmax(prediction)
        confidence = prediction[0][self.predicted_idx]
        

    def print_pose(self):
        """
    Prints the pose name or None if no person detected.
    """   
        self.class_names = [
        "TreePose",
        "DownwardDog",
        "CobraPose"]   
        self.pose_name = self.class_names[self.predicted_idx]
        
        return self.pose_name


    def release_webcam(self):
        self.camera.release()
        

        




 

