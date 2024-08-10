import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
#用于单标签
#返回yolov5生成txt中的实际点位，以及与原始图片大小相同的纯色画布
def txt2mask(img_path,txt_path):
    img = cv2.imread(img_path)  #读取图片信息
    img_x = img.shape[0]
    img_y = img.shape[1]
    with open(txt_path, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
    data = data.split('\n')[0]
    d = data.split(' ',-1)
    #d[-1] = d[-1][0:-1]
    data = []
    for i in range(1,int(len(d)/2)+1):
        data.append([img_y * float(d[2*i-1]),img_x * float(d[2*i])])
    data.append(data[0])
    data = np.array(data, dtype=np.int32) 
 
    img = np.zeros((img_x,img_y,1)) #白色背景
 
    return data,img

#txt文件夹操作
img_dir = ''
txt_dir = 'data_unl/auto_annotate_labels_sam_b'
save_dir = 'data_unl/masks'
files = os.listdir(img_dir)
for file in files:
    print(file)
    name = file[0:-4]
    img_path = img_dir + '/' + name + '.jpg'
    txt_path = txt_dir + '/' + name + '.txt'
    if os.path.exists(txt_path) == False:
        # print(txt_path)
        continue
    data,img = txt2mask(img_path,txt_path)
    color = 255
    cv2.fillPoly(img,   # 原图画板
             [data], # 多边形的点
             color=color)
    # save_path = save_dir + '/' + name + '.png'
    # cv2.imwrite(save_path, img)
    # crop image based on mask
    image = cv2.imread(img_path)
    mask = img
    mask_crop = mask.copy()
    mask = mask.astype(np.float32) / 255.
    image = image * mask
    mask = mask.astype(np.uint8)  
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        crop = mask_crop[y-10:y+h+10, x-10:x+w+10]
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(save_dir + '/' + name + '.jpg', crop)

