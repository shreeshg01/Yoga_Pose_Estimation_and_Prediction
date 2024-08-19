import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time

pred = ""

def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True 
    return False

def main():
    global pred

    model = load_model("model.h5")
    label = np.load("labels.npy")

    holistic = mp.solutions.pose
    holis = holistic.Pose()
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while True:
        lst = []

        _, frm = cap.read()

        height, width, _ = frm.shape
        window = np.zeros((height, width * 2, 3), dtype="uint8")

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        frm = cv2.blur(frm, (4, 4))
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            start_time = time.time()  # Start timing

            p = model.predict(lst)
            pred = label[np.argmax(p)]

            accuracy = p[0][np.argmax(p)]

            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time

            print("Detected Pose:", pred)
            print("Accuracy:", accuracy)
            print("Time to Recognize Pose:", elapsed_time, "seconds")

            

            if accuracy > 0.75:
                cv2.putText(window, pred, (int(width * 1.5), 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)

                # Display corresponding image based on the predicted pose
                
                
                    
                # here in this code we have to give the image path 
                if pred == "Standing relax":
                    pose_image1 = cv2.imread("Goddess1.jpg")
                    pose_image2 = cv2.imread("Star1.jpg")
                    pose_image3 = cv2.imread("Halfmoon1.jpg")
                    pose_image4 = cv2.imread("Tree1.jpg")
                    # Display three different images for Goddess1
                    resized_pose_image1 = cv2.resize(pose_image1, (int(width / 2), int(height / 2)))
                    resized_pose_image2 = cv2.resize(pose_image2, (int(width / 2), int(height / 2)))
                    resized_pose_image3 = cv2.resize(pose_image3, (int(width / 2), int(height / 2)))
                    resized_pose_image4 = cv2.resize(pose_image4, (int(width / 2), int(height / 2)))
                    cv2.imshow("Goddess1.jpg", resized_pose_image1)
                    cv2.imshow("Star1.jpg", resized_pose_image2)
                    cv2.imshow("Halfmoon1.jpg", resized_pose_image3)
                    cv2.imshow("Tree1.jpg", resized_pose_image4)
                    cv2.waitKey(5000)
                    
                    cv2.destroyWindow("Goddess1.jpg")
                    cv2.destroyWindow("Star1.jpg")
                    cv2.destroyWindow("Halfmoon1.jpg")
                    cv2.destroyWindow("Tree1.jpg")
                    # Continue to display other poses
                    continue
                elif pred == "Sitting relax":
                    pose_image1 = cv2.imread("Butterfly1.jpg")
                    pose_image2 = cv2.imread("Sideplank1.jpg")
                    pose_image3 = cv2.imread("Sukhasan1.jpg")
                    
                    # Display three different images for Goddess1
                    resized_pose_image1 = cv2.resize(pose_image1, (int(width / 2), int(height / 2)))
                    resized_pose_image2 = cv2.resize(pose_image2, (int(width / 2), int(height / 2)))
                    resized_pose_image3 = cv2.resize(pose_image3, (int(width / 2), int(height / 2)))
                    resized_pose_image4 = cv2.resize(pose_image4, (int(width / 2), int(height / 2)))
                    cv2.imshow("Butterfly1.jpg", resized_pose_image1)
                    cv2.imshow("Sideplank1.jpg", resized_pose_image2)
                    cv2.imshow("Sukhasan1.jpg", resized_pose_image3)
                    
                    cv2.waitKey(5000)
                    
                    cv2.destroyWindow("Goddess1.jpg")
                    cv2.destroyWindow("Sideplank1.jpg")
                    cv2.destroyWindow("Sukhasan1.jpg")
                    
                    # Continue to display other poses
                    continue
                elif pred == "Butterfly1":
                    pose_image = cv2.imread("Butterfly2.jpeg")
                elif pred == "Butterfly2":
                    pose_image = cv2.imread("Sitting relax.jpg")
                elif pred == "Sukhasan1":
                    pose_image = cv2.imread("Sukhasan2.jpg")
                elif pred == "Sukhasan2":
                    pose_image = cv2.imread("Sitting relax.jpg")
                elif pred == "Sideplank1":
                    pose_image = cv2.imread("Sideplank2.jpg")
                elif pred == "Sideplank2":
                    pose_image = cv2.imread("Sitting relax.jpg")
                elif pred == "Goddess1":
                    pose_image = cv2.imread("Goddess2.jpg")
                elif pred == "Goddess2":
                    pose_image = cv2.imread("Standing relax.jpg")
                elif pred == "Star1":
                    pose_image = cv2.imread("Star2.jpg")
                elif pred == "Star2":
                    pose_image = cv2.imread("Standing relax.jpg")
                elif pred == "Halfmoon1":
                    pose_image = cv2.imread("Halfmoon2.jpg")
                elif pred == "Halfmoon2":
                    pose_image = cv2.imread("Standing relax.jpg")
                
                elif pred == "Tree1":
                    pose_image = cv2.imread("Tree2.jpg")
                elif pred == "Tree2":
                    pose_image = cv2.imread("Tree3.jpg")
                elif pred == "Tree3":
                    pose_image = cv2.imread("Standing relax.jpg")  
                else:
                    pose_image = None
                

                if pose_image is not None:
                     resized_pose_image = cv2.resize(pose_image, (int(width / 2), int(height / 2)))  # Adjust scaling factor as needed
                     cv2.imshow("next Pose Image", resized_pose_image)


            else:
                cv2.putText(window, "Asana is not trained", (int(width * 1), 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)

        else: 
            cv2.putText(frm, "Make Sure Full body is visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                               connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6), 
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

        window[:, :width, :] = cv2.resize(frm, (width, height))
        
        cv2.imshow("window", window)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break

if __name__ == "__main__":
    main()

