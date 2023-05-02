import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from adam import Adam
import math
import os
from models.LSM_3D import LSM3d

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
DATA_PATH = '/data/fno/plas_N987_T20.mat'

N = 987
ntrain = 900
ntest = 80
in_channels = 1
out_channels = 4
r1 = 1
r2 = 1
s1 = int(((101 - 1) / r1) + 1)
s2 = int(((31 - 1) / r2) + 1)
t = 20

batch_size = 20
learning_rate = 0.0005
epochs = 501
step_size = 100
gamma = 0.5

num_basis = 12
num_token = 4
width = 32
patch_size = [7, 2, 2]
padding = [12, 1, 11]

model_save_path = './checkpoints/plas'
model_save_name = 'plas_lsm.pt'

################################################################
# load data and data normalization
################################################################
reader = MatReader(DATA_PATH)
x_train = reader.read_field('input')[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1, 1, 1).repeat(1, 1, s2, t, 1)
y_train = reader.read_field('output')[:ntrain, ::r1, ::r2][:, :s1, :s2]
reader.load_file(DATA_PATH)
x_test = reader.read_field('input')[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1, 1, 1).repeat(1, 1, s2, t, 1)
y_test = reader.read_field('output')[-ntest:, ::r1, ::r2][:, :s1, :s2]
print(x_train.shape, y_train.shape)
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)
y_normalizer.cuda()

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)

################################################################
# models
################################################################
model = LSM3d(in_channels, out_channels, width, patch_size, num_basis, num_token, padding).cuda()
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False, p=2)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s1, s2, t, out_channels)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
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

            out = model(x).reshape(batch_size, s1, s2, t, out_channels)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    train_reg /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, train_reg, test_l2)

    if ep % 50 == 0:
        truth = y[0].squeeze().detach().cpu().numpy()
        pred = out[0].squeeze().detach().cpu().numpy()
        ZERO = torch.zeros(s1, s2)
        truth_du = np.linalg.norm(truth[:, :, :, 2:], axis=-1)
        pred_du = np.linalg.norm(pred[:, :, :, 2:], axis=-1)

        lims = dict(cmap='RdBu_r', vmin=truth_du.min(), vmax=truth_du.max())
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 6))
        t0, t1, t2, t3, t4 = 0, 4, 9, 14, 19
        ax[0, 0].scatter(truth[:, :, 0, 0], truth[:, :, 0, 1], 10, truth_du[:, :, 0], **lims)
        ax[1, 0].scatter(pred[:, :, 0, 0], pred[:, :, 0, 1], 10, pred_du[:, :, 0], **lims)
        ax[0, 1].scatter(truth[:, :, 4, 0], truth[:, :, 4, 1], 10, truth_du[:, :, 4], **lims)
        ax[1, 1].scatter(pred[:, :, 4, 0], pred[:, :, 4, 1], 10, pred_du[:, :, 4], **lims)
        ax[0, 2].scatter(truth[:, :, 9, 0], truth[:, :, 9, 1], 10, truth_du[:, :, 9], **lims)
        ax[1, 2].scatter(pred[:, :, 9, 0], pred[:, :, 9, 1], 10, pred_du[:, :, 9], **lims)
        ax[0, 3].scatter(truth[:, :, 14, 0], truth[:, :, 14, 1], 10, truth_du[:, :, 14], **lims)
        ax[1, 3].scatter(pred[:, :, 14, 0], pred[:, :, 14, 1], 10, pred_du[:, :, 14], **lims)
        ax[0, 4].scatter(truth[:, :, 19, 0], truth[:, :, 19, 1], 10, truth_du[:, :, 19], **lims)
        ax[1, 4].scatter(pred[:, :, 19, 0], pred[:, :, 19, 1], 10, pred_du[:, :, 19], **lims)
        fig.show()
