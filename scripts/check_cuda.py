import torch
print('cuda_available', torch.cuda.is_available())
print('cuda_count', torch.cuda.device_count())
print('torch_cuda', torch.version.cuda)
