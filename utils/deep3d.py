from deep3D.models import networks
from PIL import Image
import numpy as np
import torch
import os
import os.path as osp
from deep3D.util.preprocess import align_img
import torchvision.transforms as transforms
# from retinaface.pre_trained_models import get_model

from scipy.io import loadmat, savemat

import cv2
# from retinaface.pre_trained_models import get_model



def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

def get_data_path(root='datasets/examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def get_landmarks(im_path):
    #image = cv2.imread(im_path)
    
    #image = im_path.cpu().numpy()
    #image = np.transpose(image, (1, 2, 0))
    tensor_to_pil = transforms.ToPILImage()
    image = tensor_to_pil(im_path).convert('RGB')
    #image.save("./a.jpg")
    image = np.array(image)
    
    #print("reach1")
    model = get_model("resnet50_2020-07-20", max_size=2048) 
    model.eval()
    #print("reach2")
    annotation = model.predict_jsons(image) # problematic part
    #print("reach3")



    max_score_dict = max(annotation, key=lambda x: x['score'])
    landmarks = max_score_dict['landmarks']
    return landmarks

def read_data(im_path, lm3d_std, to_tensor=True):
    # to RGB 

    imgs = []
    lms = []
    cnt = 0
    for i in range(len(im_path)): # iterates #(images) times
        img = im_path[i]
        tensor_to_pil = transforms.ToPILImage()
        im = tensor_to_pil(img).convert('RGB')
        #print("img before alignment..")
        im.save(f"/workspace/images/align/before_{cnt}.jpg")

        #im = Image.open(im_path).convert('RGB')
        W,H = im.size
        

        #lm 관련된거 다 지우기 (align은 안하지만 resize and crop만)
        lm = get_landmarks(im_path[i]) # problematic part
        lm = np.array(lm)
        #print("getting landmarks..")
        
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]

        print(f"im shape :{im.size},\nlm shape : {np.array(lm).shape},\nlm3d_std : {np.array(lm3d_std).shape}")
        
        _, im, lm, _ = align_img(im, lm, lm3d_std) # original funtion
       #im = align_img(im) 
        
        
        #print("img after alignment..")
        im.save(f"/workspace/images/align/after_{cnt}.jpg")


        if to_tensor:
            im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            lm = torch.tensor(lm).unsqueeze(0)

        imgs.append(im)
        lms.append(lm)

        cnt += 1
    
    imgs = torch.cat(imgs, dim=0)
    lms = torch.cat(lms, dim=0)
    
    return imgs, lms
    #return im, lm

def load_lm3d(bfm_folder):

    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D