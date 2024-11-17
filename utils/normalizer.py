import torch
import zarr
import numpy as np
from typing import Union, Dict
import torch.nn as nn

from replay_buffer import ReplayBuffer
class LinearNormalizer(nn.Module):
    aviliable_modes = ['limits', 'gaussian']
    def __init__(self):
        super().__init__()
        self.params_dict=nn.ParameterDict()
        
    @torch.no_grad()
    def fit(self,
            data:Union[Dict, torch.Tensor, np.ndarray, zarr.Array], 
            last_n_dims = 1, 
            dtype = torch.float32, 
            mode = 'limits',
            output_max = 1,
            output_min = -1, 
            range_eps = 1e-4,
            fit_offset = True):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value, 
                    last_n_dims = last_n_dims,
                    dtype = dtype,
                    mode = mode,
                    output_max = output_max,
                    output_min = output_min,
                    range_eps = range_eps,
                    fit_offset = fit_offset
                )
        else:
            self.params_dict['_default'] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset)
            
        # def __call__(self, x:Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        #     return self.normalize(x)
    
    
    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward = True)
    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward = False)
    
    def _normalize_impl(self, x, forward = True):
        if isinstance(x, dict):
            normalized_ret = {}
            for key, value in x.items():
                params = self.params_dict[key]
                normalized_ret[key] = _normalize(value, params, forward=forward)
                return normalized_ret
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)
        

def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    
    return x
    
def _fit(data: Union[torch.Tensor, np.ndarray, zarr.Array],
         last_n_dims = 1,
         dtype = torch.float32,
         mode = 'limits',
         output_max = 1, 
         output_min = -1, 
         range_eps = 1e-4,
         fit_offset = True):

    assert mode in ['limits', 'gaussian']
    assert last_n_dims > 0
    assert output_max > output_min
    
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)
        
    dim = 1 
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)
    
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis =0)
    
    if mode == 'limits':
        if fit_offset:
            input_range = input_max - input_min
            output_range = output_max - output_min 
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_range
             
            scale = output_range / input_range
            offset = output_min - input_min * scale
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1 
        scale = 1 / scale
        offset = -input_mean/scale
    this_params = nn.ParameterDict({
        'scale':scale,
        'offset':offset,
        'input_stats':nn.ParameterDict({
            'min':input_min,
            'max':input_max,
            'mean':input_mean,
            'std':input_std,
        })
    })
    
    for p in this_params.parameters():
        p.requires_grad_(False)
        
    return this_params

    
    
    
if __name__ == "__main__":
    replay_buffer = ReplayBuffer.create_empty_zarr('./exam_zarr')
    
    data = {
        'item1':np.random.randn(3),
        'item2':np.random.randn(33)
    }
    datan = []
    for _ in range(100):
        datan.append(data)
    data_dict = {}
    for key in datan[0].keys():
        data_dict[key] = np.stack([x[key] for x in datan])
    
    replay_buffer.add_episode(data_dict)
    
    # data_exam = replay_buffer.get_episode(2)
    data_exam = replay_buffer.get_episode_idxs()
    print(data_exam)
    # z_root = zarr.group()
    # agent_pos = z_root.create_group('agent_pos')
    # agent_pos.require_dataset('rotate',data, shape= data.shape)
    # print(z_root.tree())
    # print(data.mean(axis = 0))
    # print(data.std(axis = 0))
    # normalizer = LinearNormalizer()
    # normalizer.fit(data,mode='gaussian',last_n_dims=1)
    # datan = normalizer.normalize(data)
    # print(datan.mean(axis = 0))
    # print(datan.std(axis = 0))