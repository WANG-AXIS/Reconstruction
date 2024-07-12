import torch
import numpy as np
from deeprecon.torch.projector import projector2d

sino = np.fromfile("CT_8.raw", dtype='float32')
sino = sino.reshape((900, 1642))
sino[np.isnan(sino)] = 0
sino = sino[None,None,...]
sino = torch.from_numpy(sino).cuda()
sino = sino.flip(3)

projector = projector2d(512, 1642, 0.0089924788, 0.00625, 5, 9.5, scan_type='arc')
projector = projector.cuda()

angles =  torch.arange(0, 900) * torch.pi * 2 / 900
angles = angles.cuda()

img = projector.filtered_backprojection(sino, angles)

img = img.cpu().numpy()
img = img.squeeze()
img = img[-1::-1, -1::-1]
img = img.transpose()

img.tofile("reconCT8.raw")