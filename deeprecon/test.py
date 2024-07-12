import numpy as np
import torch
from deeprecon.torch.projector import projector2d
import matplotlib.pyplot as plt


projector = projector2d(512, 736, 0.006641, 0.01, 5.95, 10, scan_type='para')
projector = projector.cuda()

views = torch.arange(0, 736) * torch.pi * 2 / 736
views = views.cuda()

a = np.load('L333_data001.npy')
a = torch.FloatTensor(a).cuda()

a.clamp_(0,1)
a = a[None, None,:,:]


p = projector.projection(a, views)

plt.figure()
plt.imshow(p.cpu().numpy().squeeze(), cmap='gray', vmin=p.min(), vmax=p.max())
plt.show()

b = projector.projection_t(p, views)

plt.figure()
plt.imshow(b.cpu().numpy().squeeze(), cmap='gray', vmin=b.min(), vmax=b.max())
plt.show()

b2 = projector.backprojection_w(p, views)

plt.figure()
plt.imshow(b2.cpu().numpy().squeeze(), cmap='gray', vmin=b2.min(), vmax=b2.max())
plt.show()

p2 = projector.backprojection_t(a, views)

plt.figure()
plt.imshow(p2.cpu().numpy().squeeze(), cmap='gray', vmin=p2.min(), vmax=p2.max())
plt.show()

x = projector.filtered_backprojection(p, views)

plt.figure()
plt.imshow(x.cpu().numpy().squeeze(), cmap='gray', vmin=0.42, vmax=0.62)
plt.show()

print(((b-b2)**2).sqrt().mean())
print(((p-p2)**2).sqrt().mean())