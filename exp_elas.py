import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.adam import Adam
from utils.params import get_args
from model_dict import get_model
import math
import os

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
args = get_args()

PATH_Sigma = os.path.join(args.data_path, './Meshes/Random_UnitCell_sigma_10.npy')
PATH_XY = os.path.join(args.data_path, './Meshes/Random_UnitCell_XY_10.npy')
PATH_rr = os.path.join(args.data_path, './Meshes/Random_UnitCell_rr_10.npy')

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name

################################################################
# models
################################################################
model, model_iphi = get_model(args)
print(count_params(model), count_params(model_iphi))
params = list(model.parameters()) + list(model_iphi.parameters())

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
# training and evaluation
################################################################
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
    if ep % step_size == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        print('save model')
        torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))