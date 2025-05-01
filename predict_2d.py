import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from torch.cuda.amp import autocast as autocast

from segmentation.model import itunet_2d
from segmentation.utils import get_weight_path


def predict_process(test_path,config,base_dir):
    # Check if CUDA is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # get weight
    weight_path = get_weight_path(config.ckpt_path)
    print(weight_path)

    # get net
    net = itunet_2d(n_channels=config.channels,n_classes=config.num_classes, image_size= tuple((384,384)), transformer_depth = 24)
    
    # Load checkpoint
    checkpoint = torch.load(weight_path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])

    pred = []
    # Move model to the appropriate device
    net = net.to(device)
    net.eval()

    in_1 = sitk.ReadImage(os.path.join(base_dir,test_path + '_0000.nii.gz'))
    in_2 = sitk.ReadImage(os.path.join(base_dir,test_path + '_0001.nii.gz'))
    in_3 = sitk.ReadImage(os.path.join(base_dir,test_path + '_0002.nii.gz'))

    in_1 = sitk.GetArrayFromImage(in_1).astype(np.float32)
    in_2 = sitk.GetArrayFromImage(in_2).astype(np.float32)
    in_3 = sitk.GetArrayFromImage(in_3).astype(np.float32)

    image = np.stack((in_1,in_2,in_3),axis=0)

    with torch.no_grad():
        for i in range(image.shape[1]):
            new_image = image[:,i,:,:]
            for j in range(new_image.shape[0]):
                if np.max(new_image[j]) != 0:
                    new_image[j] = new_image[j]/np.max(new_image[j])
            new_image = np.expand_dims(new_image,axis=0)
            data = torch.from_numpy(new_image)

            data = data.to(device)

            if device.type == 'cuda':
                with autocast():
                    output = net(data)
            else:
                output = net(data)
                
            if isinstance(output,tuple) or isinstance(output,list):
                seg_output = output[0]
            else:
                seg_output = output  
            seg_output = torch.softmax(seg_output, dim=1).detach().cpu().numpy()   
            pred.append(seg_output) 
    
    pred = np.concatenate(pred,axis=0).transpose((1,0,2,3))
    return pred

def save_npy(
    data_path: Union[Path, str],
    ckpt_path_base: Union[Path, str] = './new_ckpt/seg',
    save_dir_base: Union[Path, str] = './segout'
):
    config = Config()
    for fold in range(1,6):
        print('****fold%d****'%fold)
        config.fold = fold
        config.ckpt_path = os.path.join(ckpt_path_base, config.version, f'fold{fold}')
        save_dir = os.path.join(save_dir_base, config.version, f'fold{fold}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(data_path)]
        pathlist = list(set(pathlist))

        print(f"Processing {len(pathlist)} files for fold {fold}")
        for i, path in enumerate(pathlist):
            print(f"Processing file {i+1}/{len(pathlist)}: {path}")
            pred = predict_process(path,config,data_path)
            print(f"Prediction shape: {pred.shape}")
            np.save(os.path.join(save_dir,path+'.npy'),pred)
        print(f"Completed fold {fold}")

def vote_dir(datadir = None):
    config = Config()
    if datadir is None:
        datadir = f'./segout/{config.version}'
    outdir = os.path.join(datadir,'avg')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    path_list = list(os.listdir(os.path.join(datadir,'fold1')))
    print(f"Voting on {len(path_list)} files")
    for i, path in enumerate(path_list):
        print(f"Voting on file {i+1}/{len(path_list)}: {path}")
        re  = np.stack([np.load(os.path.join(datadir,'fold'+str(i),path))for i in range(1,6)],axis=0)
        re = np.mean(re,axis=0)
        np.save(os.path.join(outdir,path),re)
    print("Voting completed")

def postprecess(
    outdir: Union[Path, str],
    data_dir: Optional[Union[Path, str]] = None
):
    config = Config()
    if data_dir is None:
        data_dir = f'./segout/{config.version}/avg'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    path_list = list(os.listdir(data_dir))
    label_structure = np.ones((3, 3, 3))
    print(f"Post-processing {len(path_list)} files")
    for i, path in enumerate(path_list):
        print(f"Post-processing file {i+1}/{len(path_list)}: {path}")
        temp = np.load(os.path.join(data_dir,path))
        temp = temp[1]
        temp[temp<0.5] = 0
        from report_guided_annotation import extract_lesion_candidates

        # process softmax prediction to detection map
        cspca_det_map_npy = extract_lesion_candidates(
            temp, threshold='dynamic')[0]

        # remove (some) secondary concentric/ring detections
        cspca_det_map_npy[cspca_det_map_npy<(np.max(cspca_det_map_npy)/2)] = 0

        blobs_index, num_blobs = ndimage.label(cspca_det_map_npy, structure=label_structure)
        max_b,max_s = 0,0
        temp = np.zeros(cspca_det_map_npy.shape,dtype=np.uint8)
        for lesion_candidate_id in range(num_blobs):
            s = np.sum(blobs_index == (1+lesion_candidate_id))
            if s > max_s:
                max_s = s
                max_b = lesion_candidate_id
        for lesion_candidate_id in range(num_blobs):
            if lesion_candidate_id != max_b and np.sum(cspca_det_map_npy[blobs_index == (1+lesion_candidate_id)]) <= 1000:
                cspca_det_map_npy[blobs_index == (1+lesion_candidate_id)] = 0

        blobs_index, num_blobs = ndimage.label(cspca_det_map_npy, structure=label_structure)
        
        temp[cspca_det_map_npy>0.5] = 1
        print(f"Shape: {temp.shape}, dtype: {temp.dtype}")
        print(f"Sum: {np.sum(temp)}, num_blobs: {num_blobs}")
        np.save(os.path.join(outdir,path),temp)
    print(f"Post-processing completed for {len(path_list)} files")
            
class Config:    
    input_shape = (384,384)
    channels = 3
    num_classes = 2

    version = 'itunet_d24'
    fold = 1
    ckpt_path = f'./new_ckpt/seg/{version}/fold{str(fold)}'

if __name__ == '__main__':
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU for inference")

    data_path = '/users/aca21pv/ITUNet-for-PICAI-2022-Challenge/output/nnUNet_test_data'

    if not os.path.exists(data_path):
        print(f"Warning: {data_path} doesn't exist.")
        data_path = '/users/aca21pv/ITUNet-for-PICAI-2022-Challenge/segmentation/dataset/segdata/data_3d'
        print(f"Trying alternative path: {data_path}")
        if not os.path.exists(data_path):
            print(f"Error: {data_path} doesn't exist either.")
            print("Please specify a valid data path with test files.")
            exit(1)
            
    outdir = './segout/segmentation_result'
    # Create main output directory if it doesn't exist
    if not os.path.exists(outdir):
        print(f"Creating output directory: {outdir}")
        os.makedirs(outdir, exist_ok=True)
    
    segout_base = './segout'
    if not os.path.exists(segout_base):
        print(f"Creating base output directory: {segout_base}")
        os.makedirs(segout_base, exist_ok=True)
    
    print("Starting save_npy operation")        
    save_npy(data_path)
    print("save_npy completed")
    
    config = Config()
    version_dir = f'./segout/{config.version}'
    if not os.path.exists(version_dir):
        print(f"Creating version directory: {version_dir}")
        os.makedirs(version_dir, exist_ok=True)
    
    print("Starting vote_dir operation")
    vote_dir()
    print("vote_dir completed")
    
    print("Starting postprocess operation")
    postprecess(outdir)
    print("All operations completed successfully")
