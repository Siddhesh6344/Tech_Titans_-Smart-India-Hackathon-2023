import PySimpleGUI as sg
from ultralytics import YOLO
import cv2
from postprocessing import *
import pymongo
import datetime
import numpy as np
import pickle
import smtplib
import ssl
import pandas as pd
import numpy as np
import datetime
# Establish a connection to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")  

# Create or access a database
mydb = client["Steel_Plant_Ladles"]  

# Create or access a collection
mycollection = mydb["Ladles"] 

new_collection=mydb['ladle_extraction']
def extract_ts(ser,start,end):
    # print(ser[start:end])
    max_lable=ser[start:end].mode()
    # print(max_lable[0])
    return max_lable[0]

def extract_all_ladles():
    
    # Partitioning the series based on indices
    all_docs=list(mycollection.find())
    df=pd.DataFrame.from_records(all_docs)
    df.drop(["_id"],axis=1)
    ser=df['label']
    ser_loc=ser[ser == -1]
    ser_loc.index

    # Making the raw dict
    all_lables=[]
    start_time=[]
    end_time=[]
    print("ser_loc.index = ",ser_loc.index)
    print("length of ser_loc.index = ",len(ser_loc.index))

    if(len(ser_loc.index)!=0):
        for i,inx in enumerate(ser_loc.index[:-1]):
            if(ser_loc.index[i+1]>ser_loc.index[:-1][i]+1):
                print("inx, i = ",i," ",inx)
                print("extract_ts = ",extract_ts(ser,ser_loc.index[i]+1,ser_loc.index[i+1]))
                all_lables.append(extract_ts(ser,ser_loc.index[i]+1,ser_loc.index[i+1]))
                start_time.append(df['img_read'][ser_loc.index[i]+1])
                end_time.append(df['img_read'][ser_loc.index[i+1]-1])
    else:
        modeval=extract_ts(ser,0,ser.size)
        all_lables.append(modeval)
        start_time.append(df['img_read'][0])
        end_time.append(df['img_read'][ser.size-1])
        
    # Converting the dict to dataframe
    mydict={"start_time":start_time, "end_time":end_time, "labels":all_lables}
    finaldf=pd.DataFrame(mydict)
    finaldf['total_time_min']=(finaldf['end_time']-finaldf['start_time']).apply(datetime.timedelta.total_seconds)/60
    # finaldf = finaldf[~(finaldf['total_time_min'] <= 0.01)]
    # for i in range(finaldf.shape[0]):
    # finaldf['total_time_min'][i]=datetime.timedelta.total_seconds(finaldf['end_time'][i]-finaldf['start_time'][i])
    cols=finaldf.columns
    
    # Extracting records in mongodb supported format
    all_records=[]
    for i in range(finaldf.shape[0]):
        tempdict={}
        for j,col in enumerate(cols):
            print(i)
            print(j)
            print(finaldf.iloc[i,j])
            print(col)
            tempdict[col]=finaldf.iloc[i,j]    
        all_records.append(tempdict)

    # All records uploaded in the collection
    for i,record in enumerate(all_records):
        new_collection.insert_one(record)

    return 

def sendnot(msg):
    smtp_port = 587
    smtp_server = "smtp.gmail.com"
    
    message = "Problem with ladle"+ str(msg)
    
    email_from = "rijumukherjee12345678@gmail.com"
    email_to = "raj.enggentre@gmail.com"
    
    pswd = "ppcrcevrgltpncgs"
    simple_email_context = ssl.create_default_context()
    try:
        # Connect to the server
        print("Connecting to server...")
        TIE_server = smtplib.SMTP(smtp_server, smtp_port)
        TIE_server.starttls(context=simple_email_context)
        TIE_server.login(email_from, pswd)
        print("Connected to server :-)")
        print()
        print(f"Sending email to - {email_to}")
        TIE_server.sendmail(email_from, email_to, message)
        print(f"Email successfully sent to - {email_to}")


    except Exception as e:
        print(e)

    finally:
        TIE_server.quit()

# Create Layouy of the GUI
layout = [  
            [sg.Text('GUI Object Detection with Yolo V8')],
            [sg.Text('Enter Model Name'), sg.InputText(default_text="best.pt",key='model_name')],
            [sg.Text('Scale to Show'), sg.InputText(default_text="100", key= 'scale_percent')],
            [sg.Text('Emergency Email'), sg.InputText(default_text="100", key= 'thresh')],
            [sg.Button('Run'), sg.Button('Stop'), sg.Button('Close')],
            [sg.Image(filename='', key='image')],[sg.Text("Temperature of environment", size=(0, 1), key='OUTPUT2')]
            ]

# Create the Window #
window = sg.Window('GUIYoloV8', layout, finalize=True)
run_model = False
emergency = 0
# Event Loop to process "events"
while True:
    event, values = window.read(timeout=1)
    # When press Run
    if event == 'Run' : 
        # Set up model and parameter
        model = YOLO("best.pt")
        class_list = model.model.names
        scale_show = int(values['scale_percent'])
        # Read Video
         # Read Video
        #video = cv2.VideoCapture(0)
        URL = "http://192.168.232.38"
        AWB = True
        video = cv2.VideoCapture(URL + ":81/stream")
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
            raw_class_id=np.array(results[0].boxes.cls)
            print("raw_class_id = ",raw_class_id)
            print("type of raw_class_id = ",type(raw_class_id))
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
            emergency=class_id
            # if(save_img): mydict['img']=pymongo.binary.Binary( pickle.dumps( frame, protocol=2) )
            inserted_doc = mycollection.insert_one(mydict)

        else: 
            # Break the loop if not read
            video.release()
            run_model = False

extract_all_ladles()

client.close()
# Close window
window.close()