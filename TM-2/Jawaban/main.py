import os
import cv2
import numpy as np

# Load haar cascade
# Ensure the xml file is in the same directory or provide full path
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Global variables
face_recognizer = None
person_names = []
model_path = './face_recognizer_model.xml'

# Paths (using local paths like main.py for convenience)
train_path = './images/train'
test_path = './images/test'

def train_and_test():
    global face_recognizer, person_names
    
    # --- Step 1: Training ---
    print("\n--- Start Training ---")
    
    if not os.path.exists(train_path):
        print(f"Error: Train path '{train_path}' not found.")
        return

    face_list = []
    class_list = []
    
    # Get list of names from train folder
    # We filter to ensure we only get directories, ignoring hidden files
    person_names = [name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]

    for idx, name in enumerate(person_names):
        person_img_path = os.path.join(train_path, name)

        for image_name in os.listdir(person_img_path):
            full_img_path = os.path.join(person_img_path, image_name)
            
            # Read image in grayscale
            img_gray = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
            
            # Skip unreadable images
            if img_gray is None:
                continue

            detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            
            if len(detected_faces) < 1:
                continue
            
            for rect in detected_faces:
                x, y, w, h = rect
                face_img = img_gray[y:y+w, x:x+h]
                
                face_list.append(face_img)
                class_list.append(idx)

    if len(face_list) == 0:
        print("No faces found in training data.")
        return

    # Create & train LBPH model
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list, np.array(class_list))
    face_recognizer.save(model_path)
    print("Training Completed and model saved.")

    # --- Step 2: Testing ---
    print("\n--- Start Testing for Accuracy ---")
    
    if not os.path.exists(test_path):
        print(f"Error: Test path '{test_path}' not found.")
        return

    total_prediction = 0
    correct_prediction = 0
    
    test_folder_names = os.listdir(test_path)
    
    for true_name in test_folder_names:
        # Robust ID Matching (from index.py): 
        # Find the ID based on the name text, not just the folder order.
        if true_name in person_names:
            true_id = person_names.index(true_name)
        else:
            # If a name in 'test' isn't in 'train', skip it
            continue
            
        person_test_path = os.path.join(test_path, true_name)
        
        if not os.path.isdir(person_test_path):
            continue
            
        for img_name in os.listdir(person_test_path):
            full_img_path = os.path.join(person_test_path, img_name)
            img_gray = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
            
            if img_gray is None:
                continue

            detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            
            if len(detected_faces) < 1:
                continue
            
            for rect in detected_faces:
                x, y, w, h = rect
                face_img = img_gray[y:y+w, x:x+h]
                
                # Predict
                predicted_label, confidence = face_recognizer.predict(face_img)
                
                if predicted_label == true_id:
                    correct_prediction += 1
                
                total_prediction += 1

    if total_prediction > 0:
        accuracy = (correct_prediction / total_prediction) * 100
        print(f"Average Accuracy: {accuracy:.2f}%")
    else:
        print("No valid test images found.")

def predict_custom_image():
    global face_recognizer, person_names

    # Check if model is loaded/trained
    if face_recognizer is None:
        if os.path.exists(model_path):
            # Try to load existing model file if user skipped step 1
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_recognizer.read(model_path)
            
            # We need to reload names if we just restarted the script
            if len(person_names) == 0 and os.path.exists(train_path):
                 person_names = [name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]
        else:
            print("Model not trained yet. Please choose option 1 first.")
            return

    path = input("Enter the absolute path to the image: ")
    
    if not os.path.exists(path):
        print("Error: File not found.")
        return

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print("Error: Cannot read image.")
        return
        
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

    if len(detected_faces) < 1:
        print("No face detected.")
        return

    for rect in detected_faces:
        x, y, w, h = rect
        face_img = img_gray[y:y+w, x:x+h]
        
        label, confidence = face_recognizer.predict(face_img)
        
        # Safety check if label is within range
        if label < len(person_names):
            name = person_names[label]
        else:
            name = "Unknown"

        text = f"{name} ({confidence:.2f})"
        
        # Draw green box and text
        cv2.rectangle(img_bgr, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img_bgr, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
    cv2.imshow('Result', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main_menu():
    while True:
        print("\n=== Face Recognition System ===")
        print('1. Train and test model')
        print('2. Predict single image')
        print('3. Exit')
        
        choice = input('Enter your choice: ')
        
        if choice == '1':
            train_and_test()
        elif choice == '2':
            predict_custom_image()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid input, please try again.")

if __name__ == "__main__":
    main_menu()