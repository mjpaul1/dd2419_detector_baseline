#!/usr/bin/env python

import numpy as np
import utils
import PIL
import json

from PIL import Image

def main():

    annFile="./dd2419_coco/annotations/training.json"
    with open(annFile) as json_file: 
        data = json.load(json_file)

    for image in range(0,len(data["annotations"])):
        #get image path
        imgFile = data["images"][image]['file_name']
        imgFile_path = './dd2419_coco/training/'+imgFile
        #print(image)
        img = Image.open(imgFile_path)

        #a = 1
        #b = 0   
        #c = 5 #left/right (i.e. 5/-5)
        #d = 0
        #e = 1
        #f = 0 #up/down (i.e. 5/-5)
        #img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)
        x = data["annotations"][image]["bbox"][0]
        y = data["annotations"][image]["bbox"][1]
        w = data["annotations"][image]["bbox"][2]
        h = data["annotations"][image]["bbox"][3]
        x_center = x + w / 2.0
        y_center = y + h / 2.0

        upper = y_center - 100
        left = x_center - 133
        lower = y_center + 100
        right = x_center + 133
        img = img.crop((left, upper, right, lower))
        img = img.resize((640,480))

        #black and white converting
        #im = np.array(Image.open(imgFile_path).convert('L'))

        #new dict for image info
        nb=1258+image
        imgdict = {"id":nb , "width": 640, "height": 480, "file_name": "cr_"+imgFile}

        #add it in data
        data["images"].append(imgdict)

        #new dict for bbox info
        bboxdict = data["annotations"][image]
        bboxdict["id"]=nb
        bboxdict["image_id"]=nb
        bboxdict["bbox"]=[0, 0, w, h]

        #add it in data
        data["annotations"].append(bboxdict)

        #change info in data
        data['info']["version"]=2.0
        data['info']["description"]='DD2419 traffic sign dataset augmented with cropped images'

        #save new file 
        #gr_im= Image.fromarray(im).save("./dd2419_coco/training/tr_"+imgFile)
        img.save("./dd2419_coco/training/cr_"+imgFile)

        #save new .json annotation file
        with open("./dd2419_coco/annotations/augmented.json", "w") as outfile:  
            json.dump(data, outfile) 

    
    #im = np.array(Image.open('./dd2419_coco/training/000000.jpg').convert('L'))
    #gr_im= Image.fromarray(im).save('gr_000000.jpg')



    #img = Image.open('./dd2419_coco/training/000000.jpg')
    #flip_1 = np.fliplr(img)
    # TensorFlow. 'x' = A placeholder for an image.
    height = 480.0
    width = 640.0
    #shape = [height, width, channels]
    #x = tf.placeholder(dtype = tf.float32, shape = shape)
    #flip_2 = tf.image.flip_up_down(x)
    #flip_3 = tf.image.flip_left_right(x)
    #flip_4 = tf.image.random_flip_up_down(x)
    #flip_5 = tf.image.random_flip_left_right(x)

    #utils.save(flip_1, './augm/000000.jpg')








if __name__ == "__main__":
    main()