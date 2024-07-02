import pickle
import numpy as np
from typing import Mapping, Dict, Hashable
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform
from utility import *

class LoadPreprocessed(MapTransform):
    """
    import preprocessed data.
    output mmap_mode numpy array
    """

    def __init__(
        self,
        keys: KeysCollection, 
        optional_keys: KeysCollection = None,
        mmap_mode = True
    ) -> None: 
        self.keys = keys
        self.optional_keys = optional_keys
        self.mmap_mode = mmap_mode

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        basename = os.path.basename(d[self.first_key(d)])
        basename = basename.split('.')[0]
        with open(join(os.path.dirname(d[self.first_key(d)]), basename+'.pkl'), 'rb') as f:
            d['image_meta_dict'] = pickle.load(f)
        if 'label' in self.keys:
            basename = os.path.basename(d['label'])
            basename = basename.split('.')[0]
            if isfile(join(os.path.dirname(d['label']), basename+'.pkl')):
                with open(join(os.path.dirname(d['label']), basename+'.pkl'), 'rb') as f:
                    d_label = pickle.load(f)
                    d['image_meta_dict']['labels'] =  str(d_label['labels'])
        for key in self.keys:
            if self.mmap_mode:
                d[key] = np.load(d[key], mmap_mode='r')
            else:
                d[key] = np.load(d[key])
        if self.optional_keys != None:
            for key in self.optional_keys:
                if key in d.keys():
                    basename = os.path.basename(d[key])
                    basename = basename.split('.')[0]
                    pkl_name = join(os.path.dirname(d[key]), basename+'.pkl')
                    if isfile(pkl_name):
                        with open(pkl_name, 'rb') as f:
                            d_label = pickle.load(f)
                            if 'labels' in d_label.keys():
                                d['image_meta_dict']['labels'] =  str(d_label['labels'])
                    if self.mmap_mode:
                        d[key] = np.load(d[key], mmap_mode='r')
                    else:
                        d[key] = np.load(d[key])
                else:
                    d[key] = None
        return d
    