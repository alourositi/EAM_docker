import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_3D_coordinates(u, v, d, intrin, pos, quat):
    
    fx = intrin[0]
    fy = intrin[1]
    cx = intrin[2]
    cy = intrin[3]
    
    X = d / fx * (u - cx)
    Y = d / fy * (v - cy)
    Z = d

    vec = np.array([X,Y,Z])
    r = R.from_euler('zyx', [0, 0, -90], degrees=True).as_matrix()
    vec = np.dot(r,vec)

    r_q = R.from_quat(quat)
    vec = np.dot(r_q,vec)

    vec = vec + np.array(pos)

    return [vec[0], vec[1], vec[2]]
    
class Object:
    
    def __init__(self, type, score, bbox, id, depth, intrin, pos, quat):
        self.obj_type = type
        self.score = score
        self.bbox = bbox
        self.obj_id = id
        self.width = self.bbox[2]- self.bbox[0]
        self.height = self.bbox[3]- self.bbox[1]

        self.min=get_3D_coordinates(self.bbox[0], self.bbox[1], depth, intrin, pos, quat)
        self.max=get_3D_coordinates(self.bbox[2], self.bbox[3], depth, intrin, pos, quat)
        

    def __repr__(self):
        return f'Detection(ID = {self.obj_id}, type = {self.obj_type}, score = {self.score}, [x1 y1 x2 y2] = {self.bbox})'

    def draw_detection(self,image):
       image = cv2.rectangle(image, (int(self.bbox[0]),int(self.bbox[1])), (int(self.bbox[2]),int(self.bbox[3])), (36,255,12), 2)
       #cv2.putText(image, f'{self.obj_type}: {str(round(self.score, 2))}, id: {self.obj_id}', (int(self.bbox[0])+2,int(self.bbox[1] + 17)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
       cv2.putText(image, f'{self.obj_type}: {str(round(self.score, 2))}', (int(self.bbox[0])+2,int(self.bbox[1] + 17)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
       

    def get_center_coords(self):
        return ((self.bbox[0]+self.bbox[2])//2,(self.bbox[1]+self.bbox[3])//2)

    def update_id(self, id):
        self.obj_id = id
        
class Person(Object):

    def __init__(self, type, score, bbox, id, depth, intrin, pos, quat):
        super().__init__(type, score, bbox, id, depth, intrin, pos, quat)
        if (self.width > 1.5*self.height):
            self.is_victim = True
            self.obj_type = "Victim"
            self.is_responder = False
        elif (self.height > 1.5*self.width):
            self.is_responder = True
            self.obj_type = "Responder"
            self.is_victim = False
        else:
            self.is_victim = False
            self.is_responder = False

            
class Vehicle(Object):
    def __init__(self, type, category, score, bbox, id, depth, intrin, pos, quat):
        super().__init__(type, score, bbox, id, depth, intrin, pos, quat)
        self.category = category
	
	
class Drone(Object):
    def __init__(self, type, score, bbox, det_id, depth, intrin, pos, quat):
        super().__init__(type, score, bbox, det_id, depth, intrin, pos, quat)
        if (self.max[2] > 1.5):
            self.airborne = True
        else:
            self.airborne = False

	    
	    
	    
	    
	    
	    
	    
	    
