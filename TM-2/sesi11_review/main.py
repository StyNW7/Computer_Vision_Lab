#minioreo.notion.site/computer-vision
import os
import cv2
import math
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

train_path = 'images/train'
test_path = 'images/test'
model_path = 'face_recognizer_model.xml'

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def menu1():
    person_name = os.listdir(train_path)
    face_list = []
    class_list = []

    for idx, name in enumerate(person_name):
        image_path = os.path.join(train_path, name)
        for image_name in os.listdir(image_path):
            full_img_path = os.path.join(image_path, image_name)
            img_gray = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
            detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            
            if len(detected_faces) < 1:
                continue
            
            for rect in detected_faces:
                x, y, w, h = rect
                face_img = img_gray[y:y+w, x:x+h]
                
                face_list.append(face_img)
                class_list.append(idx)

    face_recognizer.train(face_list, np.array(class_list))
    face_recognizer.save(model_path)

    #average accuracy prediction
    total_prediction = 0
    correct_prediction = 0

    classes = os.listdir(train_path)
    for label, person_name in enumerate(classes):
        person_test_path = os.path.join(test_path, person_name)
        
        for img_name in os.listdir(person_test_path):
            full_img_path = os.path.join(person_test_path, img_name)
            img_gray = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
            detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            
            if len(detected_faces) < 1:
                continue
            
            for rect in detected_faces:
                x, y, w, h = rect
                face_img = img_gray[y:y+w, x:x+h]
                predicted_label, _ = face_recognizer.predict(face_img)
                
                if predicted_label == label:
                    correct_prediction += 1
                
                total_prediction += 1

    print("Average Accuracy: ", (correct_prediction/total_prediction * 100))

#predict
def menu2(img_path):
    face_recognizer.read(model_path)
    test_img = img_path

    img_bgr = cv2.imread(test_img)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    person_names = os.listdir(train_path)
    detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

    for rect in detected_faces:
        x, y, w, h = rect
        face_img = img_gray[y:y+w, x:x+h]
        label, confidence = face_recognizer.predict(face_img)
        
        name = person_names[label]
        confidence = math.floor(confidence*100)/100
        text = name + ' ' + str(confidence)
        
        cv2.rectangle(img_bgr, (x,y), (x+w, y+h), (0,255,0), 1)
        cv2.putText(img_bgr, text, (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
        
        cv2.imshow('result', img_bgr)
        cv2.waitKey(0)


def print_menu():
    print('1. Train and test model')
    print('2. Predict image')

def main_menu():
    choice = -1
    while(True):
        print_menu()
        choice = int(input('Enter your choice: '))
        
        if choice == 1:
            menu1()
        elif choice == 2:
            path = input('Enter the absolute path: ')
            menu2(path)

main_menu()

# menu2('D:/QuizPrep/images/test/robert/1.jpg')
        
    


#test
# test_dir = 'images/test'
# classes = os.listdir(test_dir)
# for index, person_name in enumerate(classes):
#     class_path = train_dir + '/' + person_name
#     for image_path in os.listdir(class_path):
#         full_img_path = class_path + '/' + image_path
#         img_bgr = cv2.imread(full_img_path)
#         img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        
#         if len(detected_faces) < 1:
#             continue
        
#         for rect in detected_faces:
#             x, y, w, h = rect
#             face_img = img_gray[y:y+w, x:x+h]
            
#             res, confidence = face_recognizer.predict(face_img)
#             confidence = math.floor(confidence * 100) / 100
#             cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 1)
#             text = classes[res] + ' ' + str(confidence) + '%'
#             cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
            
#             cv2.imshow('result', img_bgr)
#             cv2.waitKey(0)


# for image_path in os.listdir(test_dir):
#     full_img_path = test_dir + '/' + image_path
#     img_bgr = cv2.imread(full_img_path)
#     img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
#     detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        
#     if len(detected_faces) < 1:
#         continue
    
#     for rect in detected_faces:
#         x, y, w, h = rect
#         face_img = img_gray[y:y+w, x:x+h]
            
#         res, confidence = face_recognizer.predict(face_img)
#         confidence = math.floor(confidence * 100) / 100
        
#         cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 1)
#         text = classes[res] + ' ' + str(confidence) + '%'
#         cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
        
#         cv2.imshow('result', img_bgr)
#         cv2.waitKey(0)