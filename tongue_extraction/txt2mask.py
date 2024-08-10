import os
import cv2
import numpy as np
 
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
 
#txt单文件测试
# img_path = 
# txt_path = 
# data,img = txt2mask(img_path,txt_path)
# color = 128
# cv2.fillPoly(img,   # 原图画板
#              [data], # 多边形的点
#              color=color)
# cv2.imwrite('', img)
 
#txt文件夹操作
img_dir = ''
txt_dir = r'\data_unl\auto_annotate_labels_sam_b'
save_dir = 'data_unl/masks'
files = os.listdir(img_dir)
for file in files:
    name = file[0:-4]
    img_path = img_dir + '/' + name + '.jpg'
    txt_path = txt_dir + '/' + name + '.txt'
    data,img = txt2mask(img_path,txt_path)
    color = 255
    cv2.fillPoly(img,   # 原图画板
             [data], # 多边形的点
             color=color)
    save_path = save_dir + '/' + name + '.png'
    cv2.imwrite(save_path, img)
    