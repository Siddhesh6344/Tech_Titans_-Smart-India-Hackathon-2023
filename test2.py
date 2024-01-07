import PySimpleGUI as sg
from ultralytics import YOLO
import cv2
from postprocessing import *
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import datetime
import numpy as np
import pickle

uri = "mongodb+srv://ujjawalrr:try@steelplantladles.kwrxlx6.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))

mydb = client['SteelPlantLadles']
save_img=False
mycollection=mydb['Ladles']
# mycollection = mydb["mycollection"]

# Create Layouy of the GUI
layout = [  
            [sg.Text('GUI Object Detection with Yolo V8')],
            [sg.Text('Enter Model Name'), sg.InputText(default_text="best.pt",key='model_name')],
            [sg.Text('Scale to Show'), sg.InputText(default_text="100", key= 'scale_percent')],
            [sg.Button('Run'), sg.Button('Stop'), sg.Button('Close')],
            [sg.Image(filename='', key='image')]
            ]

# Create the Window #
window = sg.Window('GUIYoloV8', layout, finalize=True)
run_model = False
# Event Loop to process "events"
while True:
    event, values = window.read(timeout=1)
    # When press Run
    if event == 'Run' :
        # Set up model and parameter
        model = YOLO(r"local_host\best.pt")
        class_list = model.model.names
        scale_show = int(values['scale_percent'])
        # Read Video
        video = cv2.VideoCapture(0)
        # Run Signal
        run_model = True
    # When press Stop or close window or press Close
    elif event in ('Stop', sg.WIN_CLOSED, 'Close'):
        if run_model :
            run_model = False # Stop running
            video.release() # Release video
            window['image'].update(filename='') # Destroy picture
        # When close window or press Close
        if event in (sg.WIN_CLOSED, 'Close'): break
    # Run Model
    if run_model :
        ret, frame = video.read()
        t_read=datetime.datetime.now()
        if ret :
            results = model.predict(frame)
            t_label=datetime.datetime.now()
            raw_class_id=results[0].boxes.cls
            if(len(raw_class_id)==0):
                class_id=-1
            elif(len(raw_class_id)==1):
                class_id=int((raw_class_id).flatten())
            else:
                class_id=0
                for i in range(len(raw_class_id)):
                    class_id*=10
                    class_id+=raw_class_id.flatten()[i]
            labeled_img = draw_box(frame, results[0], class_list)
            display_img = resize_image(labeled_img, scale_show)
            # Show Image
            imgbytes = cv2.imencode('.png', display_img)[1].tobytes()
            window['image'].update(data=imgbytes)
            mydict = {"label": class_id, "img_read":t_read, "img_label":t_label}
            if(save_img): mydict['img']=pymongo.binary.Binary( pickle.dumps( frame, protocol=2) )
            inserted_doc = mycollection.insert_one(mydict)

        else:
            # Break the loop if not read
            video.release()
            run_model = False

client.close()
# Close window
window.close()