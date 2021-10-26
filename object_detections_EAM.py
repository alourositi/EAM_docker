import cv2
import socket
import struct
import threading
import json 
import time
#import netifaces as ni
import numpy as np
import os

from utils.eucl_tracker import EuclideanDistTracker
from modules.detector.predictor import COCODemo
from kafka import KafkaProducer
from utils.objects import Person
from utils.functions import*
from utils.frame_message import Frame
#import multiprocessing
import multiprocessing as mp

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

def handle_client(Client, address,q):
    
    # Read messages
    while True:

        data = bytearray()
        new_message = True
        payload = 18

        while True:

            msg = Client.recv(payload-len(data))
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

        # Create frame from messages
        current_frame = Frame(bytes(data))

        # Push to queue
        q.put(current_frame)
        print("Frame pushed to queue")

def run_detections(frame_q,det_q):

    #Initializations
    detector = COCODemo(min_image_size=640, confidence_threshold=0.9)
    #tracker = EuclideanDistTracker()
    #producer = KafkaProducer(bootstrap_servers=['195.251.117.126:9091'],
    #                         value_serializer=lambda x:
    #                        json.dumps(x).encode('utf-8'))
    producer = KafkaProducer(bootstrap_servers=[str(os.environ['IP_KAFKA']) + ':' + str(os.environ['PORT_KAFKA'])],
                             value_serializer=lambda x:
                            json.dumps(x).encode('utf-8'))


    start_time = time.time()
    unique_dets=[]
    counter_uniq_object_id=0
    last_uniq_object_id=0

    while True:

        if not frame_q.empty():
            
            frame = frame_q.get()
            
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
                    obj = Person(result_labels[x],result_scores[x],result_bboxes[x],x,z)
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

                det_q.put([frame.image,frame.depth,frame.CameraId])
                print("Detection pushed to queue")

        if (time.time() - start_time) > 9:
            print("10 seconds passed")
            
            if unique_dets!=[]:
                # Send new detections over Kafka
                if last_uniq_object_id==0:
                    kafka_thread = threading.Thread(name='non-daemon', target=generates_msg(unique_dets,producer))
                    kafka_thread.start()
                    print("Sent detections to Kafka(1st)")
                    last_uniq_object_id= counter_uniq_object_id
                else:
                    if unique_dets[last_uniq_object_id:]!=[]:
                        kafka_thread = threading.Thread(name='non-daemon', target=generates_msg(unique_dets[last_uniq_object_id:],producer))
                        kafka_thread.start()
                        print("Sent detections to Kafka")
                        last_uniq_object_id= counter_uniq_object_id
            
            #unique_dets = [] #Set list of detection object as empty eath time send object to kafka
            start_time = time.time()
        

        

                    
                    
                    ## Track detections
                    #tracked_detections = tracker.update(detections)
#
#
                    #if tracked_detections!=[]:
                    #    for t in tracked_detections:
                    #        # Display detections
                    #        t.draw_detection(frame.image)
#
                    #        if t.obj_id not in unique_ids:
                    #            #time.sleep(1)
                    #            # Send new detections over Kafka
                    #            kafka_thread = threading.Thread(name='non-daemon', target=generates_msg(t,producer))
#
                    #            kafka_thread.start()
                    #            #print("Sent detection message to kafka.")
                    #            unique_ids.append(t.obj_id)
                    
                    
                    #det_q.put(frame.image)
                    


def main():

    # Initialize TCP server

    ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #host= ni.ifaddresses('eno1')[ni.AF_INET][0]['addr'] # Return the IP of server
    #port = 4567
    host= '0.0.0.0' #Get IP of EAM from .env file 
    port = int(os.environ['PORT_EAM']) #Get port of EAM from .env file

    try:
        ServerSocket.bind((host, port))
        print("Listening for connections")
    except socket.error as e:
        print(str(e))

    # Initialize queues and processes

    frame_q = mp.Queue()
    det_q = mp.Queue()

    detector_process = mp.Process(target=run_detections, args=(frame_q, det_q))
    #display_process=mp.Process(target=display_frame, args=(det_q,))
    detector_process.start()
    #display_process.start()

    # Listen for connections

    ServerSocket.listen()

    try:
        while True:
            Client, address = ServerSocket.accept()
            client_process = mp.Process(target=handle_client, args=(Client, address,frame_q))
            client_process.start()
    except KeyboardInterrupt: 
        client_process.join()
        detector_process.join()
        #display_process.join()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main() 
