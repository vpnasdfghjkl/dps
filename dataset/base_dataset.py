import torch
from typing import Dict

class BaseImageDataset(torch.utils.data):
    def get_validation_dataset(self) -> 'BaseImageDataset':
        return BaseImageDataset
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, id: int) -> Dict[str, torch.Tensor]:
        """_summary_

        Args:
            id (int): _description_

        Returns:
            Dict[str, torch.Tensor]:= 
            {
                obs:{
                    img01: (T, C, H, W) # rgb
                    img02: (T, C, H, W) # rgb
                    ...
                    state_vector: (T, D_state: int = 128) # low_dim
                    ...
                    pcd: (T, 512, 3) # pcd
                    ...
                },
                action: (T, D_a)
            }
        """
        raise NotImplementedError()