#tensorflow 2+
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from flask import Flask, render_template
import RPi.GPIO as GPIO
import glob
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

os.system('modprobe w1-gpio')
os.system('modprobe w1-therm')

base_dir = '/sys/bus/w1/devices/'
device_folder = glob.glob(base_dir + '28*')[0]
device_file = device_folder + '/w1_slave'

path = 'NanumBarunGothic.ttf'
fontprop =fm.FontProperties(fname=path, size=18)

facenet = cv2.dnn.readNet('model/deploy.prototxt', 'model/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('model/mask_detector.model')
#model.summary()

def read_temp_raw():
    f = open(device_file, 'r')
    lines = f.readlines()
    f.close()
    return lines

def read_temp():
    lines = read_temp_raw()
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = read_temp_raw()
    equals_pos = lines[1].find('t=')
    if equals_pos != -1:
        temp_string = lines[1][equals_pos+2:]
        temp_c = float(temp_string) / 1000.0
        return temp_c

def mask_predict():
    img = cv2.imread('./static/img/capture.jpg')
    h, w = img.shape[:2]

    plt.figure(figsize=(8, 5))
    plt.imshow(img[:, :, ::-1])


    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    facenet.setInput(blob)
    dets = facenet.forward()

    faces = []

    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:
            continue

        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        face = img[y1:y2, x1:x2]
        faces.append(face)

    #plt.figure(figsize=(16, 6))

    for i, face in enumerate(faces):
        plt.subplot(1, len(faces), i + 1)
        plt.imshow(face[:, :, ::-1])

    #plt.figure(figsize=(16, 5))

    for i, face in enumerate(faces):
        face_input = cv2.resize(face, dsize=(224, 224))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)

        mask, nomask = model.predict(face_input).squeeze()

        plt.subplot(1, len(faces), i + 1)
        plt.imshow(face[:, :, ::-1])
        if (mask * 100) < 30 :
            plt.title('마스크를 미착용했습니다. %.2f%%' % (100 - mask * 100), fontproperties=fontprop)
            print("red_led on")
            GPIO.output(17, True)
        else :
            plt.title('마스크를 착용했습니다.  %.2f%%' % (mask * 100),fontproperties=fontprop)
            print("red_led off")
            GPIO.output(17, False)
        #plt.show()
        plt.savefig('./static/img/output10.jpg')
        print("-------process loading ---------")




app = Flask(__name__)

@app.route('/')
def home():
    temp = read_temp()
    return render_template('monitering.html', temp=temp, image_file='img/output10.jpg')

if __name__ == '__main__':
    try :
        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret,frame = cap.read()
            # Display the resulting frame
            cv2.imshow('frame',frame)
            key = cv2.waitKey(10)
            if key == 27:
                break
            if key == ord(' '):
                cv2.imwrite('./static/img/capture.jpg',frame)
                mask_predict()
                app.run(host='localhost', port=8080)
    except :
        GPIO.output(17, False)

# When everything done, release the capture
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows() 
    


