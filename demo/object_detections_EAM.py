import os
import cv2
import threading
import time
import json

import pyrealsense2 as rs
import numpy as np

import math
from statistics import median
from matplotlib.pyplot import box
from numpy.lib.arraysetops import unique
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from kafka import KafkaProducer
from tracker3 import *

Fx=0
Fy=0
Cx=0
Cy=0

def get_3D_coordinates(u, v, d):
    
    X = d / Fx * (u - Cx)
    Y = d / Fy * (v - Cy)
    Z = d

    return [X, Y, Z]

class Detection:
    
    def __init__(self, label, score, bbox, det_id, depth):
        self.label = label
        self.score = score
        self.bbox = bbox
        self.det_id = det_id
        self.width = self.bbox[2]- self.bbox[0]
        self.height = self.bbox[3]- self.bbox[1]

        self.min=get_3D_coordinates(self.bbox[0], self.bbox[1], depth)
        self.max=get_3D_coordinates(self.bbox[2], self.bbox[3], depth)
        

    def __repr__(self):
        return f'Detection(ID = {self.det_id}, label = {self.label}, score = {self.score}, [x1 y1 x2 y2] = {self.bbox})'
    
    def get_center_coords(self):
        return ((self.bbox[0]+self.bbox[2])//2,(self.bbox[1]+self.bbox[3])//2)

    def set_id(self, det_id):
        self.det_id = det_id


#producer = KafkaProducer(bootstrap_servers=['195.251.117.126:9091'],
producer = KafkaProducer(bootstrap_servers=[str(os.environ['IP_KAFKA']) + ':' + str(os.environ['PORT_KAFKA'])],
                         value_serializer=lambda x: 
                         json.dumps(x).encode('utf-8'))

topic = "Object_detection"

#sources = ['color_sensor', 'depth_sensor']

def generates_msg(data): 
    
    header=[('content-encoding', b'base64'), ('type', b'detected object')]
    #msg = {
    #    'obj_id' : str(data.det_id),
    #    'bbox' : str(data.bbox),
    #    'obj_type' : str(data.label),
    #    'source' : 'EAM', 
    #    'event': 'detection'
    #    }
    msg ={
        "obj_id" : data.det_id,
        "bbox" :{
            "min" : data.min,
            "max" : data.max
        },
        "obj_type" : data.label
    }
    producer.send(topic, value=msg, headers=header)
        
def get_depth_of_center(x, y, width, height, frame):
   
    width = math.ceil(0.5 * width/2)
    height = math.ceil(0.5 * height/2)
    depths = []

    for i in range(1-width,width):    
        for j in range(1-height,height):
            depths.append(frame.get_distance(x+i,y+j))

    if len(depths)!=1:
        return median(depths)
    else:
        return depths[0]

if __name__ == "__main__":
    tracker = EuclideanDistTracker()
    config_file = "e2e_faster_rcnn_X_101_32x8d_FPN_1x_custom.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    #cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.8,
    )

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    width=640
    height=480
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 6)
    
    # Start streaming
    profile=pipeline.start(config)
    prof=profile.get_stream(rs.stream.color)

    # Get camera intrinsics
    intrin=prof.as_video_stream_profile().get_intrinsics()
    Fx=intrin.fx
    Fy=intrin.fy
    Cx=intrin.ppx
    Cy=intrin.ppy

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    print("Running detection for objects...")
    unique_ids=[]
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            
            if not aligned_depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_BONE)

            # Get detections
            result, result_labels, result_scores, result_bboxes = coco_demo.run_on_opencv_image(color_image)

            detections = []

            if result_labels!=[]:
                for x in range(len(result_labels)):
                    # Calculate depth of area around object center
                    xc = int(round(result_bboxes[x][0]+result_bboxes[x][2])//2)
                    yc = int(round(result_bboxes[x][1]+result_bboxes[x][3])//2)
                    object_width = round(result_bboxes[x][2]-result_bboxes[x][0])
                    object_height = round(result_bboxes[x][3]-result_bboxes[x][1])
                    z = get_depth_of_center(xc,yc,object_width, object_height, aligned_depth_frame)
                    # Create Detection object
                    detections.append(Detection(result_labels[x],result_scores[x],result_bboxes[x],x,z))

            # Track detections
            tracked_detections = tracker.update(detections)

            if tracked_detections!=[]:
                for t in tracked_detections:
                    time.sleep(1)
                    if t.det_id not in unique_ids:
                        # Send new detections over Kafka
                        kafka_thread = threading.Thread(name='non-daemon', target=generates_msg(t))
                        kafka_thread.start()
                        print("Sent detection message to kafka.")
                        unique_ids.append(t.det_id)

    except RuntimeError as err:
        print(err)
        pass

    finally:
        # Stop streaming
        pipeline.stop()