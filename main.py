import albumentations as A
import time
import random
import os
from utils import load_config,make_default_config,load_data,path2bboxes,path2images,save_transformed
from albu import config2pipelines
from multiprocessing import Queue,Process
from utils import init
from custom import make_gradation_images,read_gradation_images,aug_gradation


def process_func(index):
    config = load_config()
    try:
        os.makedirs(config["1_GLOBAL_OPTIONS"]["save_path"])
        os.makedirs(os.path.join(config["1_GLOBAL_OPTIONS"]["save_path"],"images"))
        os.makedirs(os.path.join(config["1_GLOBAL_OPTIONS"]["save_path"],"labels"))
    except:
        pass
    pipelines = config2pipelines()
    gradation_images = read_gradation_images(config)
    images_list,labels_list = load_data(config)
    images_list = path2images(images_list)
    labels_list = path2bboxes(labels_list)
    for img,label in zip(images_list,labels_list):
        for _ in range(config["1_GLOBAL_OPTIONS"]["augmentation_per_image"]):
            transform = pipelines[random.randint(0,len(pipelines)-1)]
            image = aug_gradation(img,config["2_AUGMENTATION"]["Gradation"]["Probability"],gradation_images)
            transformed = transform(image=image,bboxes=label)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            save_transformed(transformed_image,transformed_bboxes,config,index)
            
        
    


if __name__=="__main__":
    init()
    config = load_config()
    make_gradation_images(config)
    process_list = []
    for index in range(config["1_GLOBAL_OPTIONS"]["Process_nums"]):
        p = Process(target=process_func,args=(index,),daemon=True)
        process_list.append(p)
    [ p.start() for p in process_list]
    [ p.join() for p in process_list]
    
