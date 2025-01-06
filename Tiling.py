import openslide
import numpy as np
from PIL import Image
import cv2
from glob import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

wsi_file_list=glob("../../data/Stomach biopsy 스캔원본/*.mrxs")

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
for i in tqdm(range(len(wsi_file_list))):
    slide=openslide.OpenSlide(wsi_file_list[i])
    tile_size = int(300*(41.1/20.0))  # 각 타일 크기
    x_tiles = slide.dimensions[0] // tile_size
    y_tiles = slide.dimensions[1] // tile_size
    src_tile_size=300
    folder_name = (os.path.basename(wsi_file_list[i])).split(".")[0]
    mask = np.array(Image.open('../../data/gcu_svs/mask/'+folder_name+".png"))
    ratio=mask.shape[0]/slide.dimensions[1]
    create_dir('../../data/gcu_svs/svs/'+folder_name+"/")
    for x in range(x_tiles):
        for y in range(y_tiles):
            x_ratio_ind=[int(x*tile_size*ratio),int((x+1)*tile_size*ratio)]
            y_ratio_ind=[int(y*tile_size*ratio),int((y+1)*tile_size*ratio)]
            
            if mask[y_ratio_ind[0]:y_ratio_ind[1],x_ratio_ind[0]:x_ratio_ind[1]].mean()==0:
                continue
            region = np.array(slide.read_region((x * tile_size, y * tile_size), level=0, size=(tile_size, tile_size)))
            hsv_image = cv2.cvtColor(region[:,:,:-1], cv2.COLOR_RGB2HSV)
            ret,mask1=cv2.threshold(hsv_image[:,:,1],127,1, cv2.THRESH_OTSU)
            if ret<50:
                continue
            if mask1.sum()<tile_size*tile_size*0.5:
                continue
            Image.fromarray(region).resize((src_tile_size,src_tile_size)).save('../../data/gcu_svs/svs/'+folder_name+"/{}_{}.png".format(x,y))
