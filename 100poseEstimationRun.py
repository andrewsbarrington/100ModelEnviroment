import os
import math
import cv2
import numpy as np
import enum
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas

print("hello")

pose_list = ['Downward Dog', 'Standing Forward bend','Upward Forward Fold','Raise your arm over head-Volcano','Standing Salutation-Samasthi','Four limb staff low plank','Upward Facing Dog','Three Legged Downward Facing Dog right','Three Legged Downward Facing Dog left','Right Knee to right shoulder','Right Knee to left shoulder','Left Knee to left shoulder','left Knee to right shoulder','Chair pose','Revolved chair pose right side','Revolved chair pose left side','Warrior one - right side','Warrior one left side','Warrior two right side','Warrior two left side','Triangle pose right side','Triangle pose left side','Standing folding forward right leg scissor','Standing folding forward left leg scissor','Revolved triangle right side','Revolved triangle left side','Standing split right side','Standing split left side','High lunge right side','High lunge left side','Runners lunge right side','Runners lunge left side','Low lunge right side','Low lunge left side','Standing balance right leg raised front','Standing balance left leg raised front','Standing balance right leg raised side','Standing balance left leg raised side','Revolved low lunge right side','Revolved low lunge left side','Revolved bound high lunge right side','Revolved bound high lunge left side','Tree pose right side','Tree pose left side','Tree pose raised arm right side','Tree pose raised arm left side','Side bending tree right side','Side bending tree left side','Basic tree pose right side','Basic tree pose left side','Seated forward bend','Seated right side straddle','Seated left side straddle pose','Seated right side stretch pose','Seated left side stretch pose','Revolved seated angle right side','Revolved seated angle left side','Seated straddle head hand on ground','Child pose hand backward','Child pose hand forward','Headstand','Dolphin pose','Low cobra','Locust pose','Half bridge pose','Full bridge pose','Extended mountain backbend','Standing salut right side bend','Standing salut left side bend','Palm tree pose','Palm tree right side bend','Palm tree left side bend','Standing Shoulder stretch hand behind back','Standing yoga seal hand back upward','Plank pose','Reverse warrior right side','Reverse warrior left side','Extended side angle right side','Extended side angle left side','Elbow arm extended side angle right side','Elbow arm extended side angle left side','Bound extended side angle right side','Bound extended side angle left side','Warrior three right side','Warrior three left side','Dancer pose right side','Dancer pose left side','Half Hanumanasana right side','Half Hanumanasana left side','Upward seated straddle V pose','Shoulder stand','Plough pose','Fish pose','Supine Spinal Twist Eagle Legs right','Supine Spinal Twist Eagle Legs left','Standing eagle right side','Standing eagle left side','Mountain pose','Squat—Sitting-Down Pose','Moon pose right','Moon pose left','Corpse pose']


#initialize keras model
model = tf.keras.models.load_model('model/100PoseModel.keras')

model.summary()

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

class BodyPart(enum.Enum):
  """Enum representing human body keypoints detected by pose estimation models."""
  NOSE = 0
  LEFT_EYE = 1
  RIGHT_EYE = 2
  LEFT_EAR = 3
  RIGHT_EAR = 4
  LEFT_SHOULDER = 5
  RIGHT_SHOULDER = 6
  LEFT_ELBOW = 7
  RIGHT_ELBOW = 8
  LEFT_WRIST = 9
  RIGHT_WRIST = 10
  LEFT_HIP = 11
  RIGHT_HIP = 12
  LEFT_KNEE = 13
  RIGHT_KNEE = 14
  LEFT_ANKLE = 15
  RIGHT_ANKLE = 16

print("hello")
def detectPoseImage(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    visibilities = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        iter = 0
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            
            if(iter not in (1,3,4,6,9,10,17,18,19,20,21,22,29,30,31,32) ):
                
                #if(landmark.visibility > .4):
                landmarks.append((float(landmark.x * width), float(landmark.y * height)))

            iter += 1
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks
    

def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    print(left)
    center = tf.add(tf.multiply(left , 0.5),  tf.multiply(right, 0.5))
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

    It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                 BodyPart.RIGHT_HIP)
    
    print(BodyPart.LEFT_HIP.value)

    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                      BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                     BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size



def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
  """
  # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                 BodyPart.RIGHT_HIP)
    print(pose_center)
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    print(pose_center)
    pose_center = tf.broadcast_to(pose_center,[1, 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks


def landmarks_to_embedding(landmarks):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    #reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(tf.expand_dims(landmarks,0))
    #print(landmarks)
    
    # Flatten the normalized landmark coordinates into a vector  
    #embedding = keras.layers.Flatten()(landmarks)

    embedding = tf.reshape(landmarks,[1,34])

    return embedding


def preprocess_data(X_train):
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 51)))
        processed_X_train.append(tf.reshape(embedding, (34)))
    return tf.convert_to_tensor(processed_X_train)



# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture("yogaVideo.mp4")
camera_video.set(3,1280)
camera_video.set(4,960)

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)
df = pandas.DataFrame
out = []

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()

    timestamp_seconds = (camera_video.get(cv2.CAP_PROP_POS_MSEC))/1000

    color = (0, 0, 255)
    
    # Check if frame is not read properly.
    if not ok:
        
        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    frame, landmarks = detectPoseImage(frame, pose_video, display=False)
    
    # Check if the landmarks are detected.
    if landmarks:
        # out = []
        # for t in landmarks:
        #     for items in t:
        #         out.append(items)
        
        processedInput = landmarks_to_embedding(landmarks)
        print(processedInput)
        pose = model.predict(processedInput , batch_size=34)
        class_no = np.argmax(pose[0])
        out.append([pose_list[class_no], class_no, timestamp_seconds])
        print(pose_list[class_no])

        try:
            # Perform the Pose Classification.
            print(landmarks)
            processedInput = landmarks_to_embedding(landmarks)
            print(processedInput)
            pose = model.predict(processedInput)

            print(pose)

            # # Check if the pose is classified successfully
            # if pose != 'Unknown Pose':
            
            #     # Update the color (to green) with which the label will be written on the image.
            #     color = (0, 255, 0)
        except Exception as error:
            print(error)
    

    df = pandas.DataFrame(out).T
    
    # Display the frame.
    #cv2.putText(frame , pose , (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.imshow('Pose Classification', frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed.
    if(k == 27):
        
        # Break the loop.
        break

# Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllWindows()
