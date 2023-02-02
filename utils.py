import yaml
import os
import cv2
import time
from pathlib import Path


def init():
    main_path = (Path(__file__).parent)
    if os.path.isfile(os.path.join(main_path,'config.yaml')):
        pass
    else:
        make_default_config()
        exit()
        
def load_config():
    main_path = (Path(__file__).parent)
    with open(os.path.join(main_path,"config.yaml"),"r") as f:
        return yaml.load(f,yaml.FullLoader)
    
def load_data(config):
    images_path = config["1_GLOBAL_OPTIONS"]["images_path"]
    labels_path = config["1_GLOBAL_OPTIONS"]["labels_path"]
    images_list = os.listdir(images_path)
    labels_list = os.listdir(labels_path)
    images = []
    labels = []
    for image_name,label_name in zip(images_list,labels_list):
        images.append(os.path.join(images_path,image_name))
        labels.append(os.path.join(labels_path,label_name))
    return images,labels

def path2images(images):
    return [ cv2.imread(path) for path in images]

def path2bboxes(labels):
    new = []
    for path in labels:
        with open(path,"r") as f:
            label = f.readlines()
        label = [ line.strip().split(" ") for line in label]
        label = [ [float(line[1]),float(line[2]),float(line[3]),float(line[4]),line[0]] for line in label ]
        new.append(label)
    return new

def save_transformed(image,label,config,index):
    save_path = config["1_GLOBAL_OPTIONS"]["save_path"]
    name = str(index)+ '_'.join(str(time.time()).split("."))
    cv2.imwrite(os.path.join(save_path,"images",name+".jpg"),image)
    label = [ " ".join([str(line[4]),str(line[0]),str(line[1]),str(line[2]),str(line[3])])+"\n" for line in label]
    with open(os.path.join(save_path,"labels",name+".txt"),"w") as f:
        for line in label:
            f.write(line)
    return
        
def make_default_config():
    main_path = (Path(__file__).parent)
    config = {
        "0_Description":
            ["Total Auged Image NUMS = [Origin-Image-Nums] * [Process-Nums] * [Aug-Per-Image]"],
        "2_AUGMENTATION":{
            "RandomCrop":{
                "Probability":1,
                "Crop_size":(1280,1280),
            },
            "HorizentalFlip":{
                "Probability":0.5,
            },
            "VerticalFlip":{
                "Probability":0.5,
            },
            "Affine":{
                "Probability":0.5,
                "Translation" : 0.2,
                "Rotate":[-180,180],
            },
            "PerspectiveTransform":{
                "Probability":0.0,
                "Scale":[0.05,0.1],
            },
            "GaussianNoise":{
                "Probability":0.5,
                "Intensity":[0,35]
            },
            "Brightness":{
                "Probability":0.5,
                "Intensity":0.1,
            },
            "Contrast":{
                "Probability":0.5,
                "Intensity":0.1,
            },
            "Gamma":{
                "Probability":0.5,
                "Intensity":[40,80]
            },
            "Blur":{
                "Probability":0.5,
                "Intensity":0.2,
            },
            "Gradation":{
                "Probability":0.5,
            },
            "HueTransform":{
                "Probability":0.5,
                "Hue":15,
                "Sat":15,
            },
            "ObjectCutmix":{
                "Probability":0.5,
                "CutmixArea":[0.2,0.4],
            },
            "RandomRemove":{
                "Probability":0.5,
                "RemovePerTotal":0.2,
            },
            "RGBShift":{
                "Probability":0.5,
                "Limit":10,
            }
        },
    "1_GLOBAL_OPTIONS" : {
        "Process_nums":2,
        "augmentation_per_image":10,
        "images_path":"",
        "labels_path":"",
        "save_path":"",
        "train_test_split":True,
        "train_ratio":0.8,
        "output_image_shape_scale":0.1,
        },
    }
    with open(os.path.join(main_path,"config.yaml"),"w") as f:
        yaml.dump(config,f,default_flow_style=None,indent=4,default_style=None,line_break='key',allow_unicode=False)
