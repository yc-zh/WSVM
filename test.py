import os
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from vision_transformer.vit_model import VisionTransformer
from vision_transformer.vit_model import PatchEmbed
from matplotlib import pyplot as plt
from PIL import Image
import cv2

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

patch_size = 16
num_patches = (224 // patch_size) ** 2
edge_value = 0
TopK = 2

def get_edges(inputs):

    tensor_size =(1, 14*14)
    small_value = edge_value
    edges = torch.full(tensor_size, small_value).to(device)
    image = inputs
    imagecopy = image.copy()


    for i in range(14):
        for j in range(14):
            patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            white_pixels = np.sum(patch > 0.1)
            if white_pixels > 10 and i > 1 and not ( i < 6 and 1 < j < 12):
                edges[0, i*14+j] = 1.0
    #             x = j * patch_size
    #             y = i * patch_size            
    #             w = patch_size
    #             h = patch_size
                
    #             # 在源图像上绘制边缘部分
    #             cv2.rectangle(imagecopy, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 绘制绿色矩形
    # cv2.imshow('Edges Image', imagecopy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edges

def get_masks(mask):
    scale_ratio = 0.9
    mask_resize = cv2.resize(mask, (int(224*scale_ratio), int(224*scale_ratio)))

    left = int((224 - int(224*scale_ratio)) / 2)
    right = int(224 - int(224*scale_ratio) - left)
    mask_padding = cv2.copyMakeBorder(mask_resize, left, right, left, right, cv2.BORDER_CONSTANT, value=0)
    edge_mask = mask - mask_padding
    edge_mask[0:36, :] = 0

    return edge_mask


def test_and_visualize(model, image, image_name):
    model.eval()
    mask_dir = "/data/masks"
    with torch.no_grad():
        inputs =  image.unsqueeze(0).to(device)
        masks = Image.open(os.path.join(mask_dir, image_name)).convert('L')
        mask = np.array(masks) / 255
        edge_mask = get_masks(mask)
        edges = get_edges(edge_mask)
        outputs = model(inputs)

        # Get predictions  
        instances = outputs[:, 1:, :]
        instances_softmax = torch.softmax(instances, dim=-1)
        score = torch.mul(instances_softmax[:, :, 1], edges)

        predss = score > 0.5
        predss = torch.reshape(predss, (14, 14))
        score = torch.reshape(score, (14, 14))
        score = np.round(score.cpu().numpy(), 2)
    

        image = image.cpu().numpy().transpose(1, 2, 0)
        fig, ax = plt.subplots()

        # Display the original image
        ax.imshow(image) 

        # Draw bounding boxes around target regions
        for i in range(14):
            for j in range(14):
                if predss[i][j] == 1:
                    # Calculate bounding box coordinates
                    x = j * patch_size
                    y = i * patch_size
                    width = patch_size
                    height = patch_size

                    # Draw bounding box
                    rect = plt.Rectangle((x, y), width, height, linewidth=1, edgecolor=(0,1,0), facecolor='none')
                    ax.add_patch(rect)

        plt.axis('off')
        plt.savefig('data/scores/'+image_name, bbox_inches='tight', pad_inches=0)
        plt.show()


model_ft = VisionTransformer(img_size=224,
                            patch_size=16,
                            embed_dim=768,
                            depth=12,
                            num_heads=12,
                            representation_size=None,
                            num_classes=2,
                            drop_path_ratio=0.0,
                            drop_ratio=0.0,
                            attn_drop_ratio=0.0)

model_ft = model_ft.to(device)
model_ft = torch.load('models/', map_location=device)

# unmarked
test_dir = 'data/testset'

image_list = os.listdir(test_dir)
for image_name in image_list:
    print(image_name)
    image_path = os.path.join(test_dir, image_name)
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = data_transforms['val'](image)

    test_and_visualize(model_ft, image, image_name)
    