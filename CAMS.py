import PySimpleGUI as sg
from ultralytics import YOLO
import cv2
from postprocessing import *
# using datetime module
import datetime
import requests
import cv2
import numpy as np
import imutils
# ct stores current time
ct = datetime.datetime.now()
ts = ct.timestamp()
url = "http://192.168.232.214:8080/shot.jpg"

colwebcam1_layout = [[sg.Text("Camera View", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam1")]]
colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center')

colwebcam2_layout = [[sg.Text("Camera View GrayScale", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam2")]]
colwebcam2 = sg.Column(colwebcam2_layout, element_justification='center')
colslayout = [colwebcam1, colwebcam2]

rowfooter = [sg.Image(filename="", key="-IMAGEBOTTOM-"), [sg.Text('GUI Object Detection with Yolo V8')],
            [sg.Button('Run'), sg.Button('Stop'), sg.Button('Close')]
            ,[sg.Text("Ladle Number ,Time Stamp,Location", size=(0, 1), key='OUTPUT1')],[sg.Text("Ladle Number ,Time Stamp,Location", size=(0, 1), key='OUTPUT2')]
            ]
layout = [colslayout, rowfooter]

# Create Layouy of the GUI

# Create the Window
window = sg.Window('GUIYoloV8', layout, finalize=True)
run_model = False
# Event Loop to process "events"
while True:
    event, values = window.read(timeout=1)
    # When press Run
    if event == 'Run' : 
        # Set up model and parameter
        model = YOLO("best.pt")
        class_list = model.model.names
        scale_show = int(100)
        # Read Video
        #video = cv2.VideoCapture(0)
        URL = "http://192.168.232.38"
        AWB = True

# Face recognition and opencv setup
        video = cv2.VideoCapture(URL + ":81/stream")
        # Run Signal
        run_model = True
    # When press Stop or close window or press Close
    elif event in ('Stop', sg.WIN_CLOSED, 'Close'):
        if run_model : 
            run_model = False # Stop running
            video.release() # Release video
            #window['image'].update(filename='') # Destroy picture
        # When close window or press Close
        if event in (sg.WIN_CLOSED, 'Close'): break
    # Run Model
    if run_model : 
        ret, frame = video.read()
        if ret :
            lst=[]
            location="A"
            results = model.predict(frame)
            class_id= results[0].boxes.cls.numpy().astype(int)
            labeled_img = draw_box(frame, results[0], class_list)
            display_img = resize_image(labeled_img, scale_show)
            # Show Image
            imgbytes = cv2.imencode('.png', display_img)[1].tobytes()
            ct = datetime.datetime.now()
            
            lst.append([class_id,ct,location])
            window['cam1'].update(data=imgbytes)
            window['OUTPUT2'].update(value=lst)
        else: 
            # Break the loop if not read
            video.release()
            run_model = False
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=400, height=400) 
        lst2=[]
        location="B"
        model = YOLO("best.pt")
        results1 = model.predict(img)
        class_id= results1[0].boxes.cls.numpy().astype(int)
        labeled_img1 = draw_box(img, results1[0], class_list)
        display_img1 = resize_image(labeled_img1, scale_show)
        # Show Image
        imgbytes1 = cv2.imencode('.png', display_img1)[1].tobytes()
        ct = datetime.datetime.now()
        
        lst.append([class_id,ct,location])
        window['cam2'].update(data=imgbytes1)
        window['OUTPUT2'].update(value=lst2)
# Close window
window.close()