import torch
import numpy as np
from typing import Dict, Hashable, Mapping
from monai.transforms.transform import MapTransform
from monai.transforms import (
    MapTransform,
    ScaleIntensityRange,
)
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor

class flareScaleIntensityRanged(MapTransform):

    backend = ScaleIntensityRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler_1 = ScaleIntensityRange(a_min=990, a_max=1140, b_min=-1, b_max=1, clip=True, dtype=dtype)
        self.scaler_2 = ScaleIntensityRange(a_min=-500, a_max=750, b_min=-1, b_max=1, clip=True, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if torch.min(d[key]) > -10:
                d[key] = self.scaler_1(d[key])
            else:
                d[key] = self.scaler_2(d[key])
        return d