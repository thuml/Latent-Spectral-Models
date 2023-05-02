import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from adam import Adam
import math
import os
from models.LSM_Irregular_Geo import LSM2d
from models.LSM_Irregular_Geo import IPHI

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
PATH_Sigma = '/home/wuhaixu/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
PATH_XY = '/home/wuhaixu/elasticity/Meshes/Random_UnitCell_XY_10.npy'
PATH_rr = '/home/wuhaixu/elasticity/Meshes/Random_UnitCell_rr_10.npy'
N = 2000
ntrain = 1000
ntest = 200

in_channels = 2
out_channels = 1

batch_size = 20
learning_rate = 0.0005
epochs = 501
step_size = 100
gamma = 0.5

num_basis = 12
num_token = 4
width = 32
patch_size = [6, 6]
padding = [0, 0]
modes = 12

model_save_path = './checkpoints/elas'
model_save_name = 'elas_lsm.pt'

################################################################
# load data and data normalization
################################################################
input_rr = np.load(PATH_rr)
input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1, 0)
input_s = np.load(PATH_Sigma)
input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0).unsqueeze(-1)
input_xy = np.load(PATH_XY)
input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

train_rr = input_rr[:ntrain]
test_rr = input_rr[-ntest:]
train_s = input_s[:ntrain]
test_s = input_s[-ntest:]
train_xy = input_xy[:ntrain]
test_xy = input_xy[-ntest:]

print(train_rr.shape, train_s.shape, train_xy.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_rr, train_s, train_xy),
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_rr, test_s, test_xy),
                                          batch_size=batch_size,
                                          shuffle=False)

################################################################
# models
################################################################
model = LSM2d(in_channels, out_channels, width, patch_size, num_basis, num_token, padding).cuda()
model_iphi = IPHI().cuda()
print(count_params(model), count_params(model_iphi))

################################################################
# training and evaluation
################################################################
params = list(model.parameters()) + list(model_iphi.parameters())
optimizer = Adam(params, lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
N_sample = 1000
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for rr, sigma, mesh in train_loader:
        rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()
        samples_x = torch.rand(batch_size, N_sample, 2).cuda() * 3 - 1

        optimizer.zero_grad()
        out = model(mesh, code=rr, iphi=model_iphi)
        samples_xi = model_iphi(samples_x, code=rr)

        loss_data = myloss(out.view(batch_size, -1), sigma.view(batch_size, -1))
        loss = loss_data
        loss.backward()

        optimizer.step()
        train_l2 += loss_data.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for rr, sigma, mesh in test_loader:
            rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()
            out = model(mesh, code=rr, iphi=model_iphi)
            test_l2 += myloss(out.view(batch_size, -1), sigma.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)
