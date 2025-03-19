from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
from imutils import face_utils
import dlib

import torch
from glob import glob
# os.environ['LMDB_FORCE_CFFI'] = '1'
import lmdb



def facecrop(face_detector, face_predictor,frame_dir):

    compression = frame_dir.split('/')[-3]
    all_dirs = sorted(os.listdir(frame_dir))
    os.makedirs("./all_ldmks/full_frame_ffpp_{}_ldmk_lmdb".format(compression))
    for sub_dir in all_dirs: 
        env = lmdb.open("./all_ldmks/full_frame_ffpp_{}_ldmk_lmdb".format(compression),map_size=int(1e10))
        txn = env.begin(write=True)  
        all_frames = glob(frame_dir+'/'+sub_dir+'/*.png')
        for cnt_frame in tqdm(all_frames):
            try:
                frame_org = cv2.imread(cnt_frame)
                height,width=frame_org.shape[:-1]
                
                frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
                faces = face_detector(frame, 1)

                landmarks=[]
                size_list=[]
                if len(faces)==0:
                    continue
                for face_idx in range(len(faces)):
                    landmark = face_predictor(frame, faces[face_idx])
                    landmark = face_utils.shape_to_np(landmark)
                    x0,y0=landmark[:,0].min(),landmark[:,1].min()
                    x1,y1=landmark[:,0].max(),landmark[:,1].max()
                    face_s=(x1-x0)*(y1-y0)
                    size_list.append(face_s)
                    landmarks.append(landmark)
                landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
                landmarks=landmarks[np.argsort(np.array(size_list))[::-1]][0]
                txn.put(key = cnt_frame.encode(),
                            value = landmarks)
            except:
                print(sub_dir)
                pass
        txn.commit() 
        env.close()
    return



if __name__=='__main__':

    device=torch.device('cuda')
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    facecrop(face_detector, face_predictor, frame_dir='./FF++/c23/original/frames')
    
