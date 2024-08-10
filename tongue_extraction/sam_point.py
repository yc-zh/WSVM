from ultralytics.models.sam import Predictor as SAMPredictor
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# Create SAMPredictor
overrides = dict(task="segment", mode="predict", model="models/sam_b.pt")
predictor = SAMPredictor(overrides=overrides)

# Set image
# predictor.set_image("data/images/329.jpg")  # set with image file
# predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # set with np.ndarray
image_dir = "data_unl/data_unlabeled"

for img in os.listdir(image_dir):
    # results = None
    img_path = os.path.join(image_dir, img)
    print(img_path)
    image = cv2.imread(img_path)
    img_size = image.shape[:2]
    print(img_size)
    box_path = "runs/detect/predict/labels/" + img.split('.')[0] + '.txt'
    # box txt: 0 0.611639 0.83127 0.198954 0.263754 
    # 分别是： 0 x,y,w,h
    # 需要转换为： x1,y1,x2,y2
    box = [0, 0, 0, 0]
    with open(box_path, 'r') as f:
        box_n = [float(i) for i in f.readline().split(' ')[1:]]
        center = int(box_n[0] * img_size[1]), int(box_n[1] * img_size[0])
        w = int(box_n[2] * img_size[1])
        h = int(box_n[3] * img_size[0])
        box = [center[0] - w//2, center[1] - h//2, center[0] + w//2, center[1] + h//2]

    predictor.set_image(image)
    results = predictor(points=center, labels=[1])
    # predict
    # print(results)
    masks = results[0].masks.data.cpu().numpy()
    masks_adjusted = np.repeat(masks, 3, axis=0)
    masks_adjusted = np.transpose(masks_adjusted, (1, 2, 0))

    # 保存分割之后的mask
    mask_path = 'data_unl/test_mak/' + img.split('.')[0] + '_mask_point.png'
    cv2.imwrite(mask_path, masks_adjusted * 255)

    # 保存分割之后的image
    image_path = 'data_unl/test_mak/' + img.split('.')[0] + '_image_point.png'
    cv2.imwrite(image_path, image * masks_adjusted.astype(np.uint8))

    
    # 保存分割之后的图片
    # image_path = 'data_unl/test_mak/' + img.split('.')[0] + '_image.png'
    # cv2.imwrite(image_path, image * masks_adjusted.astype(np.uint8))
    # # Reset image
    predictor.reset_image()
    # break