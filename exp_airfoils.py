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

INPUT_X = os.path.join(args.data_path, './naca/NACA_Cylinder_X.npy')
INPUT_Y = os.path.join(args.data_path, './naca/NACA_Cylinder_Y.npy')
OUTPUT_Sigma = os.path.join(args.data_path, './naca/NACA_Cylinder_Q.npy')

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)

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
model = get_model(args)
print(count_params(model))

################################################################
# load data and data normalization
################################################################
inputX = np.load(INPUT_X)
inputX = torch.tensor(inputX, dtype=torch.float)
inputY = np.load(INPUT_Y)
inputY = torch.tensor(inputY, dtype=torch.float)
input = torch.stack([inputX, inputY], dim=-1)

output = np.load(OUTPUT_Sigma)[:, 4]
output = torch.tensor(output, dtype=torch.float)
print(input.shape, output.shape)

x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
x_test = input[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
x_train = x_train.reshape(ntrain, s1, s2, 2)
x_test = x_test.reshape(ntest, s1, s2, 2)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)

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

        optimizer.zero_grad()
        out = model(x)

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

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)

    # plot
    if ep % step_size == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        print('save model')
        torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))
        ind = -1
        X = x[ind, :, :, 0].squeeze().detach().cpu().numpy()
        Y = x[ind, :, :, 1].squeeze().detach().cpu().numpy()
        truth = y[ind].squeeze().detach().cpu().numpy()
        pred = out[ind].squeeze().detach().cpu().numpy()
        nx = 40 // r1
        ny = 20 // r2
        X_small = X[nx:-nx, :ny]
        Y_small = Y[nx:-nx, :ny]
        truth_small = truth[nx:-nx, :ny]
        pred_small = pred[nx:-nx, :ny]

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
        ax[0, 0].pcolormesh(X, Y, truth, shading='gouraud')
        ax[1, 0].pcolormesh(X, Y, pred, shading='gouraud')
        ax[2, 0].pcolormesh(X, Y, pred - truth, shading='gouraud')
        ax[0, 1].pcolormesh(X_small, Y_small, truth_small, shading='gouraud')
        ax[1, 1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud')
        ax[2, 1].pcolormesh(X_small, Y_small, np.abs(pred_small - truth_small), shading='gouraud')
        fig.show()
