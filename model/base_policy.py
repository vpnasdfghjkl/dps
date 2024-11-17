import torch.nn as nn
from typing import Dict
import torch
class BasePolicy(nn.Module):
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.tensor]:
        """_summary_

        Args:
            obs_dict (Dict[str, torch.Tensor]): _description_

        Returns:
            Dict[str, torch.tensor]: _description_
        """