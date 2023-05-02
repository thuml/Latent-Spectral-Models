import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from adam import Adam
import math
import os
from models.LSM_2D import LSM2d

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
INPUT_PATH = '/home/wuhaixu/elasticity/Interp/Random_UnitCell_mask_10_interp.npy'
OUTPUT_PATH = '/home/wuhaixu/elasticity/Interp/Random_UnitCell_sigma_10_interp.npy'
N = 2000
ntrain = 1000
ntest = 200

in_channels = 1
out_channels = 1
r1 = 1
r2 = 1
s1 = int(((41 - 1) / r1) + 1)
s2 = int(((41 - 1) / r2) + 1)

batch_size = 20
learning_rate = 0.001
epochs = 501
step_size = 100
gamma = 0.5

num_basis = 12
num_token = 4
width = 32
patch_size = [3, 3]
padding = [7, 7]

model_save_path = './checkpoints/elas_interp'
model_save_name = 'elas_interp_lsm.pt'

################################################################
# load data and data normalization
################################################################
input = np.load(INPUT_PATH)
input = torch.tensor(input, dtype=torch.float).permute(2,0,1)
output = np.load(OUTPUT_PATH)
output = torch.tensor(output, dtype=torch.float).permute(2,0,1)

x_train = input[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]

x_test = input[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]

x_train = x_train.reshape(ntrain, s1, s2, 1)
x_test = x_test.reshape(ntest, s1, s2, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)

################################################################
# models
################################################################
model = LSM2d(in_channels, out_channels, width, patch_size, num_basis, num_token, padding).cuda()
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        mask = x.clone()

        optimizer.zero_grad()
        out = model(x)
        out = out*mask

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            mask = x.clone()

            out = model(x)
            out2 = out * mask

            test_l2 += myloss(out2.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)
