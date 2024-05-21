import os
import json
import numpy as np
from PIL import Image
import torch
import copy
from torch.utils.data import Dataset
from torchvision import transforms

def get_metadata(dataset_name):
    if dataset_name == 'pascal':
        meta = {
            'num_classes': 20,
            'path_to_dataset': '/home/s/ducnq/spml-paper/data/pascal',
            'path_to_images': '/home/s/ducnq/spml-paper/data/pascal/VOCdevkit/VOC2012/JPEGImages'
        }
    elif dataset_name == 'coco':
        meta = {
            'num_classes': 80,
            'path_to_dataset': '/home/s/luongtk/VLPL/data/coco',
            'path_to_images': '/home/s/luongtk/vlm_mlc/data/coco'
        }
    elif dataset_name == 'nuswide':
        meta = {
            'num_classes': 81,
            'path_to_dataset': '/home/s/ducnq/spml-paper/data/nuswide',
            'path_to_images': '/home/s/ducnq/spml-paper/data/nuswide/Flickr/Flickr'
        }
    elif dataset_name == 'cub':
        meta = {
            'num_classes': 312,
            # 'path_to_dataset': '/home/s/ducnq/spml-paper/data/cub',
            'path_to_dataset': './data/cub',
            # 'path_to_images': '/home/s/ducnq/spml-paper/data/cub/CUB_200_2011/images',
            'path_to_images': '/kaggle/input/cub2002011/CUB_200_2011/images'
        }
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    return meta

def get_imagenet_stats():
    '''
    Returns standard ImageNet statistics. 
    '''
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    return (imagenet_mean, imagenet_std)

def get_transforms():
    '''
    Returns image transforms.
    '''
    
    (imagenet_mean, imagenet_std) = get_imagenet_stats()
    tx = {}
    tx['pl'] = transforms.Compose([
        transforms.Resize((224, 224)), #for clip size 336
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['train'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['val'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['test'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return tx

def generate_split(num_ex, frac, rng):
    '''
    Computes indices for a randomized split of num_ex objects into two parts,
    so we return two index vectors: idx_1 and idx_2. Note that idx_1 has length
    (1.0 - frac)*num_ex and idx_2 has length frac*num_ex. Sorted index sets are 
    returned because this function is for splitting, not shuffling. 
    '''
    
    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2
    
    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])
    
    return (idx_1, idx_2)

def get_data(P):
    '''
    Given a parameter dictionary P, initialize and return the specified dataset. 
    '''
    
    # define transforms:
    tx = get_transforms()
    
    # select and return the right dataset:
    if P['dataset'] == 'coco':
        ds = multilabel(P, tx).get_datasets()
    elif P['dataset'] == 'pascal':
        ds = multilabel(P, tx).get_datasets()
    elif P['dataset'] == 'nuswide':
        ds = multilabel(P, tx).get_datasets()
    elif P['dataset'] == 'cub':
        ds = multilabel(P, tx).get_datasets()
    else:
        raise ValueError('Unknown dataset.')
    
    # Optionally overwrite the observed training labels with clean labels:
    # assert P['train_set_variant'] in ['clean', 'observed']
    if P['train_set_variant'] == 'clean':
        print('Using clean labels for training.')
        ds['train'].label_matrix_obs = copy.deepcopy(ds['train'].label_matrix)
    else:
        print('Using single positive labels for training.')
    
    # Optionally overwrite the observed val labels with clean labels:
    assert P['val_set_variant'] in ['clean', 'observed']
    if P['val_set_variant'] == 'clean':
        print('Using clean labels for validation.')
        ds['val'].label_matrix_obs = copy.deepcopy(ds['val'].label_matrix)
    else:
        print('Using single positive labels for validation.')
    
    # We always use a clean test set:
    ds['test'].label_matrix_obs = copy.deepcopy(ds['test'].label_matrix)
            
    return ds

def load_data(base_path, P):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['labels_obs'] = np.load(os.path.join(base_path, 'formatted_{}_labels_obs.npy'.format(phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
        data[phase]['feats'] = np.load(P['{}_feats_file'.format(phase)]) if P['use_feats'] else []
    return data

class multilabel:

    def __init__(self, P, tx):
        
        # get dataset metadata:
        meta = get_metadata(P['dataset'])
        self.base_path = meta['path_to_dataset']
        
        # load data:
        source_data = load_data(self.base_path, P)
        
        # generate indices to split official train set into train and val:
        split_idx = {}
        (split_idx['train'], split_idx['val']) = generate_split(
            len(source_data['train']['images']),
            P['val_frac'],
            np.random.RandomState(P['split_seed'])
            )
        
        # subsample split indices: # commenting this out makes the val set map be low?
        ss_rng = np.random.RandomState(P['ss_seed'])
        temp_train_idx = copy.deepcopy(split_idx['train'])
        for phase in ['train', 'val']:
            num_initial = len(split_idx[phase])
            num_final = int(np.round(P['ss_frac_{}'.format(phase)] * num_initial))
            split_idx[phase] = split_idx[phase][np.sort(ss_rng.permutation(num_initial)[:num_final])]
        
        # define train set:
        self.train = ds_multilabel(
            P['dataset'],
            source_data['train']['images'][split_idx['train']],
            source_data['train']['labels'][split_idx['train'], :],
            source_data['train']['labels_obs'][split_idx['train'], :],
            source_data['train']['feats'][split_idx['train'], :] if P['use_feats'] else [],
            tx['train'],
            P['use_feats'],
            tx['pl'],
            True
        )
            
        # define val set:
        self.val = ds_multilabel(
            P['dataset'],
            source_data['train']['images'][split_idx['val']],
            source_data['train']['labels'][split_idx['val'], :],
            source_data['train']['labels_obs'][split_idx['val'], :],
            source_data['train']['feats'][split_idx['val'], :] if P['use_feats'] else [],
            tx['val'],
            P['use_feats']
        )
        
        # define test set:
        self.test = ds_multilabel(
            P['dataset'],
            source_data['val']['images'],
            source_data['val']['labels'],
            source_data['val']['labels_obs'],
            source_data['val']['feats'],
            tx['test'],
            P['use_feats']
        )
        
        # define dict of dataset lengths: 
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}
    
    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}

class ds_multilabel(Dataset):

    def __init__(self, dataset_name, image_ids, label_matrix, label_matrix_obs, feats, tx, use_feats, tx_pl=None, is_train=False):
        meta = get_metadata(dataset_name)
        self.num_classes = meta['num_classes']
        self.path_to_images = meta['path_to_images']
        
        self.image_ids = image_ids
        self.label_matrix = label_matrix
        self.label_matrix_obs = label_matrix_obs
        self.feats = feats
        self.tx = tx
        self.use_feats = use_feats
        self.tx_pl = tx_pl
        self.is_train = is_train

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        if self.use_feats:
            # Set I to be a feature vector:
            I = torch.FloatTensor(np.copy(self.feats[idx, :]))
        else:
            # Set I to be an image: 
            image_path = os.path.join(self.path_to_images, self.image_ids[idx])
            with Image.open(image_path) as I_raw:
                I = self.tx(I_raw.convert('RGB'))
            if self.tx_pl is not None and self.is_train == True: 
                # Split image into a 3x3 grid
                with Image.open(image_path) as I_raw:
                        height = I_raw.size[1]
                        width = I_raw.size[0]
                        
                        # Convert image to RGB if not already
                        I_raw = I_raw.convert('RGB')
                        I_clip = self.tx_pl(I_raw)
                        # Calculate the size of each tile
                        tile_width = width // 3
                        tile_height = height // 3
                        
                        # Initialize a list to store the cropped tiles
                        sub_images = []
                        sub_images.append(I_clip.unsqueeze(0))
                        # Iterate through each row and column to split the image
                        for y in range(3):
                            for x in range(3):
                                # Define the coordinates for cropping each tile
                                left = x * tile_width
                                upper = y * tile_height
                                right = left + tile_width
                                lower = upper + tile_height
                                
                                # Crop the tile from the original image
                                tile = I_raw.crop((left, upper, right, lower))
                                
                                # Apply transformations
                                tile = self.tx_pl(tile).unsqueeze(0)
                                
                                # Append the transformed tile to the list
                                sub_images.append(tile)
        
        out = {
            'image': I,
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx, :])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx, :])),
            'idx': idx,
            # 'image_path': image_path # added for CAM visualization purpose
        }
        
        if self.is_train:
            out['sub_images'] = torch.cat(sub_images) # contain original image and 9 sub-images
        
        return out

def parse_categories(categories):
    category_list = []
    id_to_index = {}
    for i in range(len(categories)):
        category_list.append(categories[i]['name'])
        id_to_index[categories[i]['id']] = i
    return (category_list, id_to_index)

def get_category_list(P):
    if P['dataset'] == 'pascal':
        catName_to_catID = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19
        }
        return {catName_to_catID[k]: k for k in catName_to_catID}
    
    elif P['dataset'] == 'coco':
        load_path = 'data/coco'
        meta = {}
        meta['category_id_to_index'] = {}
        meta['category_list'] = []

        with open(os.path.join(load_path, 'annotations', 'instances_train2014.json'), 'r') as f:
            D = json.load(f)

        (meta['category_list'], meta['category_id_to_index']) = parse_categories(D['categories'])
        return meta['category_list']

    elif P['dataset'] == 'nuswide':
        pass # TODO
    
    elif P['dataset'] == 'cub':
        pass # TODO