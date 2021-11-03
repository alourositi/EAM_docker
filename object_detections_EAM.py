from logging import exception
import cv2
import socket
import struct
import threading
import json 
import time
import os
import numpy as np
import multiprocessing as mp
import uuid

from utils.eucl_tracker import EuclideanDistTracker
from modules.detector.predictor import COCODemo
from kafka import KafkaProducer
from utils.objects import Person
from utils.functions import*
from utils.frame_message import Frame


def display_frame(det_q):
    while True:
        if not det_q.empty():
            frame=det_q.get()

            image=frame[0]
            depth=frame[1]
            CameraId= str(frame[2].hex())
            # Display RGB and Depth frame (for test purposes)
            colormap_image=cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_BONE)
            images=np.hstack((image,colormap_image))
            cv2.namedWindow('RGB + Depth '+CameraId,cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB + Depth '+CameraId,images)
            key= cv2.waitKey(1)
            if key & 0xFF == ord('q') or key==27:
                cv2.destroyAllWindows()
                break

def handle_client(s,q):

    ClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = True

    try:        
        ClientSocket.connect(s)
    except socket.error:
        print("Could not connect to " + str(s[0]))
        connected = False
        ClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while not connected:
            try:
                ClientSocket.connect(s)
                connected=True
                print("Connection established")
            except socket.error:
                time.sleep(2)
                print("Trying to connect to " + str(s[0]))
    

    # Read messages
    while True:

        data = bytearray()
        new_message = True
        payload = 18

        while True:
            try:
                msg = ClientSocket.recv(payload-len(data))
                if len(msg)==0:
                    print("Empty message")
                    print("Connection lost with " + str(s[0]))
                    connected=False
                    new_message = True
                    ClientSocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    while not connected:
                        try:
                            ClientSocket.connect(s)
                            connected=True
                            print("Reconnected to " + str(s[0]))
                        except socket.error:
                            time.sleep(2)
                            print("Trying to reconnect to " + str(s[0]))

                if new_message:

                    if msg[:1].hex()!='a5' or msg[1:2].hex()!='5a':
                       continue

                    payload = struct.unpack('l',msg[2:10])[0] + 18
                    data.extend(msg)
                    new_message= False
                    continue

                data.extend(msg) 
                if len(data)>=payload:
                    break
            except socket.error:
                print("Connection lost with " + str(s[0]))
                connected=False
                new_message = True
                ClientSocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                while not connected:
                    try:
                        ClientSocket.connect(s)
                        connected=True
                        print("Reconnected to " + str(s[0]))
                    except socket.error:
                        time.sleep(2)
                        print("Trying to reconnect to " + str(s[0]))

        # Create frame from messages
        current_frame = Frame(bytes(data))

        # Push to queue
        q.put(current_frame)
        print("Frame pushed to queue")

def run_detections(frame_q,det_q):

    #Initializations
    #detector = COCODemo(min_image_size=640, confidence_threshold=0.9)
    #producer = KafkaProducer(bootstrap_servers=[str(os.environ['IP_KAFKA']) + ':' + str(os.environ['PORT_KAFKA'])],
    #                         value_serializer=lambda x:
    #                        json.dumps(x).encode('utf-8'))
    detector = COCODemo(min_image_size=640, confidence_threshold=0.9)
    producer = KafkaProducer(bootstrap_servers=["195.251.117.126:9091"],
                             value_serializer=lambda x:
                            json.dumps(x).encode('utf-8'))

    sender_id = str(uuid.uuid4())

    start_time = time.time()
    unique_dets=[]
    counter_uniq_object_id=0
    last_uniq_object_id=0

    while True:

        if not frame_q.empty():
            
            frame = frame_q.get()
            print(frame.depth.shape)
            
            result, result_labels, result_scores, result_bboxes = detector.run_on_opencv_image(frame.image)
            detections = []
            if result_labels!=[]:
                for x in range(len(result_labels)):

                    # Calculate depth of area around object center
                    xc = int(round(result_bboxes[x][0]+result_bboxes[x][2])//2)
                    yc = int(round(result_bboxes[x][1]+result_bboxes[x][3])//2)
                    object_width = round(result_bboxes[x][2]-result_bboxes[x][0])
                    object_height = round(result_bboxes[x][3]-result_bboxes[x][1])
                    z = get_depth_of_center(xc,yc,object_width, object_height, frame.depth)
                    if z==0: # do not calculate object to close on camera
                        continue

                    # Create Detection object
                    obj = Person(result_labels[x],result_scores[x],result_bboxes[x],x,z,[frame.fx,frame.fy,frame.cx,frame.cy],[frame.px,frame.py,frame.pz], [frame.qx,frame.qy,frame.qz, frame.qw])
                    isVictim(int(result_bboxes[x][0]),int(result_bboxes[x][1]),int(result_bboxes[x][2]),int(result_bboxes[x][3]), frame.depth, obj, z)
                    if result_labels[x]=="person":
                        detections.append(obj)

            if detections!=[]:
                if unique_dets!=[]:
                    for det in detections:
                        det.draw_detection(frame.image)
                        exists = False
                        for uniq in unique_dets:
                            if calculate_spatial_distance(det,uniq) < 0.2:
                                exists = True
                                break
                        if not exists:
                            det.update_id(counter_uniq_object_id)
                            counter_uniq_object_id +=1
                            unique_dets.append(det)
                else:
                    for det in detections:
                        det.draw_detection(frame.image)
                        det.update_id(counter_uniq_object_id)
                        counter_uniq_object_id +=1
                        unique_dets.append(det)

            #det_q.put([frame.image,frame.CameraId])
            #det_q.put([frame.depth,frame.CameraId])
            det_q.put([frame.image,frame.depth,frame.CameraId])

        if (time.time() - start_time) > 9:
            print("10 seconds passed")
            
            if unique_dets!=[]:
                # Send new detections over Kafka
                if last_uniq_object_id==0:
                    kafka_thread = threading.Thread(name='non-daemon', target=generates_msg(unique_dets,producer,sender_id))
                    kafka_thread.start()
                    #print("Sent detections to Kafka(1st)")
                    last_uniq_object_id= counter_uniq_object_id
                else:
                    if unique_dets[last_uniq_object_id:]!=[]:
                        kafka_thread = threading.Thread(name='non-daemon', target=generates_msg(unique_dets[last_uniq_object_id:],producer,sender_id))
                        kafka_thread.start()
                        #print("Sent detections to Kafka")
                        last_uniq_object_id= counter_uniq_object_id
            
            #unique_dets = [] #Set list of detection object as empty eath time send object to kafka
            start_time = time.time()



def main():

    # Initialize queues and processes
    display = True
    frame_q = mp.Queue()
    det_q = mp.Queue()

    detector_process = mp.Process(target=run_detections, args=(frame_q, det_q))
    detector_process.start()
    if display:
        display_process=mp.Process(target=display_frame, args=(det_q,))
        display_process.start()
    
    # Initialize TCP server

    socks = []
    procs = []
    
    host_list = os.getenv('SERVER_IPS').split(',')
    port_list = os.getenv('SERVER_PORTS').split(',')
    for i in range(len(host_list)):
        socks.append((host_list[i],int(port_list[i])))
    
    #host= "192.168.43.47" #Get IP of EAM from .env file 
    #port = 4567 #Get port of EAM from .env file
    
    socks.append((host,port))

    #host= "195.251.117.249"
    #port= 4567
    #socks.append((host,port))
    
    
    for i in range(len(socks)):
        procs.append(mp.Process(target=handle_client, args=(socks[i], frame_q)))
        procs[i].start()
        #client_process = mp.Process(target=handle_client, args=(s, frame_q))
        #client_process.start()

    #client_process.join()
    for p in procs:
        p.join()
    detector_process.join()
    display_process.join()
    cv2.destroyAllWindows()

    
    
    
    #ClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #host= os.environ['IP_REC_FROM'] #Get IP of EAM from .env file 
    #port = int(os.environ['PORT_EAM']) #Get port of EAM from .env file
    
    #ServerSocket.bind((host, port))
    #ClientSocket.connect((host, port))
    #print(ClientSocket)
    #client_process = mp.Process(target=handle_client, args=(host, port,frame_q))
    #client_process.start()
    #client_process.join()
    #ClientSocket.close()
    #except socket.error as e:
    #    print(str(e))
    #    
    #    continue
#
    #    #except socket.error as e:
        

if __name__=="__main__":

    main() 
    
    #client_process.join()
    #detector_process.join()
    #if display:
    #    display_process.join()
    #cv2.destroyAllWindows()
