import cv2
import numpy as np

face_data = cv2.CascadeClassifier("data.xml")
camera = cv2.VideoCapture(0)

def face_detec(frame):
    optimisasi_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face = face_data.detectMultiScale(optimisasi_frame, scaleFactor=1.1)
    return face

def finish_frame():
    camera.release()
    cv2.destroyAllWindows()
    exit()
    

def box(frame):
    for x, y, w, h in face_detec(frame):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
    
    pass

def main():
    
    while True:
            _, frame = camera.read()
            box(frame)
            cv2.imshow("face detection", frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
               finish_frame()
    

if __name__ == '__main__':
    main()