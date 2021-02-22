import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)

        ran = torch.rand(10) * 0.001 + 0.03
        print('££', ran.size())
        print('$$', ran)
        ran = ran.unsqueeze(0)
        self.fc1.weight = torch.nn.Parameter(ran)
        nn.init.constant(self.fc1.bias, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        #   x = self.sigmoid(x)
        return x


#
# def real_f(x):
#     return torch.sigmoid(x[0] * .7 + x[1] * .2 + x[2] * .3 + x[3] * .4 + x[4] * .9)

def real_f(x):
    return torch.sigmoid(x[0] * .7 + x[1] * .2 + x[2] * .3 + x[3] * .4 + x[4] * .9
                         + x[5] * .7 + x[6] * .2 + x[7] * .3 + x[8] * .4 + x[9] * .9)


def real_l(x):
    return x[0] * .7 + x[1] * .2 + x[2] * .3 + x[3] * .4 + x[4] * .9 + x[5] * .7 + x[6] * .2 + x[7] * .3 + x[8] * .4 + \
           x[9] * .9


nono = Net()
criterion = torch.nn.L1Loss()
optimizer = optim.SGD(nono.parameters(), lr=.001, momentum=0.9)
nono.train()

for i in range(100000):

    optimizer.zero_grad()
    x = torch.rand(10)
    y = nono(x)
    loss = criterion(y, real_l(x))

    loss.backward()
    optimizer.step()

    for param in nono.parameters():
        print(param.data)
    print('loss', i, loss)
