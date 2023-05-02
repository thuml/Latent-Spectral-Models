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
TRAIN_PATH = '/data/fno/NavierStokes_V1e-5_N1200_T20.mat'
TEST_PATH = '/data/fno/NavierStokes_V1e-5_N1200_T20.mat'

ntrain = 1000
ntest = 200
N = 1200
in_channels = 10
out_channels = 1
r1 = 1
r2 = 1
s1 = int(((64 - 1) / r1) + 1)
s2 = int(((64 - 1) / r2) + 1)
T_in = 10
T_out = 10
step = 1

batch_size = 20
learning_rate = 0.0005
epochs = 501
step_size = 100
gamma = 0.5

num_basis = 12
num_token = 4
width = 64
patch_size = [4, 4]
padding = [0, 0]

model_save_path = './checkpoints/ns'
model_save_name = 'ns_lsm.pt'

################################################################
# load data and data normalization
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain, ::r1, ::r2, :T_in]
train_u = reader.read_field('u')[:ntrain, ::r1, ::r2, T_in:T_in + T_out]

test_a = reader.read_field('u')[-ntest:, ::r1, ::r2, :T_in]
test_u = reader.read_field('u')[-ntest:, ::r1, ::r2, T_in:T_in + T_out]

print(train_u.shape)
print(test_u.shape)

train_a = train_a.reshape(ntrain, s1, s2, T_in)
test_a = test_a.reshape(ntest, s1, s2, T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size,
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
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T_out, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T_out / step), train_l2_full / ntrain,
          test_l2_step / ntest / (T_out / step),
          test_l2_full / ntest)
