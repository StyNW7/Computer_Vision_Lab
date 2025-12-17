import os
import cv2
import numpy as np

# Load haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_recognizer = None
person_name = []

def train_and_test():
    global face_recognizer, person_name
    
    # Step 1 : Training
    print("Start Training...")
    face_list = []
    class_list = []
    
    train_path = 'dataset/train'

    # Ambil daftar nama dari folder train
    person_name = os.listdir(train_path)

    for idx, name in enumerate(person_name):
        full_path = train_path + '/' + name
        
        for img_name in os.listdir(full_path):
            img_full_path = full_path + '/' + img_name
            img = cv2.imread(img_full_path, 0) # Grayscale

            detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
            
            if len(detected_face) < 1:
                continue

            for face_rect in detected_face:
                x, y, h, w = face_rect
                face_img = img[y:y+h, x:x+w]
                face_list.append(face_img)
                class_list.append(idx)

    # Create & train model LBPH
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list, np.array(class_list))
    print("Training Completed.")
    
    # Step 2 : Testing
    print("Start Testing for Accuracy...")
    test_path = 'dataset/test'

    total_images = 0
    correct_predictions = 0
    
    test_names = os.listdir(test_path)

    for true_name in test_names:
        # Cari index id dari nama folder ini
        if true_name in person_name:
            true_id = person_name.index(true_name)
        else:
            continue

        full_path = test_path + '/' + true_name
        
        for img_name in os.listdir(full_path):
            img_full_path = full_path + '/' + img_name
            img_gray = cv2.imread(img_full_path, 0)

            detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            
            for face_rect in detected_face:
                x, y, h, w = face_rect
                face_img = img_gray[y:y+h, x:x+w]
                
                # Predict
                res_id, confidence = face_recognizer.predict(face_img)
                
                # Cek hasil prediksi id dengan real id
                if res_id == true_id:
                    correct_predictions += 1
                
                # Totalin gambar yang ada
                total_images += 1

    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"Average Accuracy: {accuracy:.2f}%") # 
    else:
        print("No faces detected in test set.")

def predict_custom_image():
    if face_recognizer is None:
        print("Model not trained yet. Please choose option 1 first.")
        return

    path = input("Enter the path to the image for testing (Absolute Path): ")
    
    if not os.path.exists(path):
        print("Error: File not found.")
        return
        
    img_bgr = cv2.imread(path)
    img_gray = cv2.imread(path, 0)
    
    if img_bgr is None:
        print("Error: Cannot read image.")
        return

    detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
    
    if len(detected_face) < 1:
        print("No face detected.")
        return

    for face_rect in detected_face:
        x, y, h, w = face_rect
        face_img = img_gray[y:y+h, x:x+w]
        
        res, confidence = face_recognizer.predict(face_img)
        
        # Gambar bounding box & text
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{person_name[res]} : {confidence:.2f}%"
        cv2.putText(img_bgr, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
    cv2.imshow('Result', img_bgr)
    cv2.waitKey(0)

# Main Program Menu
while True:
    print("\n=== Face Recognition Menu ===")
    print("1. Train and test model") 
    print("2. Predict image")         
    print("3. Exit")
    
    choice = input("Enter your choice: ")
    
    if choice == '1':
        train_and_test()
    elif choice == '2':
        predict_custom_image()
    elif choice == '3':
        print("Exiting...")
        break
    else:
        print("Invalid choice.")