import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, classification_report, f1_score

class mlp(torch.nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, n_pre_layers, n_connected_layers):
        super().__init__()
        self.n_pre_layers = n_pre_layers+1
        self.pre_channels = [int((i/(self.n_pre_layers-1))*in_channels+
                                    ((self.n_pre_layers-i-1)/(self.n_pre_layers-1))*middle_channels)
                                    for i in reversed(range(self.n_pre_layers))]
        self.pre_layers = []
        for i in range(self.n_pre_layers-1):
            self.pre_layers.append(Linear(self.pre_channels[i], self.pre_channels[i+1]))
        self.pre_layer = ModuleList(self.pre_layers)

        self.n_connected_layers = n_connected_layers+1
        self.connected_channels = [int((i/(self.n_connected_layers-1))*middle_channels+
                                    ((self.n_connected_layers-i-1)/(self.n_connected_layers-1))*out_channels)
                                    for i in reversed(range(self.n_connected_layers))]
        self.connected_layers = []
        for i in range(self.n_connected_layers-1):
            self.connected_layers.append(Linear(self.connected_channels[i], self.connected_channels[i+1]))
        self.connected_layer = ModuleList(self.connected_layers)

    def forward(self, input, dropout):
        x = input
        for i in range(self.n_pre_layers-1):
            x = self.pre_layer[i](x)
            x = x.relu()
            x = F.dropout(x, p=dropout, training=self.training)
        out_1 = x
        for i in range(self.n_connected_layers-1):
            x = self.connected_layer[i](x)
            x = x.relu()
            if i != self.n_connected_layers-2:
                x = F.dropout(x, p=dropout, training=self.training)
        out_2 = x

        return out_1, out_2

def train_mlps(model, data_loader, gt, dropout, criterion, optimizer, device):
    model.train()
    model = model.to(device)

    true_label_trn = []
    pred_label_trn = []
    training_loss = []

    for data, label in zip(data_loader, gt):
        data = data.to(device)
        # print(data.size())
        label = label.type(torch.LongTensor)
        label = label.to(device)
        pred_feat, pred_label = model(data, dropout)
        # pred_label = torch.reshape(pred_label, (2, -1))
        # print(data.size(), pred_label.size(), pred_label[0, :], label.size(), label)
        # loss = criterion(pred_label[0, :], label)
        loss = criterion(pred_label, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss = loss.item()
        training_loss.append(running_loss)
        # true_label_trn += [label.item()]
        true_label_trn += label
        pred_label_trn += pred_label.argmax(dim=1)

    # return training_loss, true_label_trn, pred_label_trn

def eval_mlps(model, data_loader, gt, dropout, criterion, device):
    model.eval()
    model = model.to(device)

    true_label_test = []
    pred_label_test = []
    testing_loss = []
    correct = 0
    test_acc = 0

    test_tmp_flag = True

    with torch.no_grad():
        for data, label in zip(data_loader, gt):
            data = data.to(device)
            if test_tmp_flag:
                data_tmp = data
                test_tmp_flag = False
            else:
                data_tmp =torch.cat((data_tmp, data), dim=0)
            label = label.type(torch.LongTensor)
            label = label.to(device)
            pred_feat, pred_label = model(data, dropout)
            test_acc += (torch.argmax(pred_label, 1).flatten() == label).type(torch.float64).mean().item()
            pred = pred_label.argmax(dim=1)
            correct += int((pred == label).sum())
            # loss = criterion(pred_label[0, :], label)
            loss = criterion(pred_label, label)

            running_loss = loss.item()
            testing_loss.append(running_loss)
            # true_label_test += [label.item()]
            true_label_test += label
            pred_label_test += pred_label.argmax(dim=1)
        # print(data_tmp, data_tmp.size())

        f1_test = f1_score(torch.tensor(true_label_test).cpu(), torch.tensor(pred_label_test).cpu(),
                           average=None, zero_division=0)
        precision_test = precision_score(torch.tensor(true_label_test, device='cpu'),
                                         torch.tensor(pred_label_test, device='cpu'),
                                         average=None, zero_division=0)
        recall_test = recall_score(torch.tensor(true_label_test, device='cpu'),
                                   torch.tensor(pred_label_test, device='cpu'),
                                   average=None, zero_division=0)
        overall_report = classification_report(torch.tensor(true_label_test, device='cpu'),
                                               torch.tensor(pred_label_test, device='cpu'),
                                               zero_division=0)

        return correct, testing_loss, test_acc, true_label_test, pred_label_test, \
               f1_test, precision_test, recall_test, overall_report

def test_mlps(model, data_loader, dropout, device, data_tmp):
    model.eval()
    model = model.to(device)

    # data_tmp = data_loader[0].to(device)
    data_tmp = data_tmp.to(device)
    pred_label_test = []

    with torch.no_grad():
        for n, data in enumerate(data_loader):
            data = data.to(device)
            pred_feat, pred_label = model(data, dropout)
            name_tmp = torch.zeros(1, 2, dtype=torch.float).to(device)
            pred_feat = torch.cat((name_tmp, pred_feat), dim=1)
            data_tmp = torch.cat((data_tmp, pred_feat), dim=0)
            pred_label_test += pred_label.argmax(dim=1)
            if n == 0:
                pred_test = pred_label
            else:
                pred_test = torch.cat((pred_test, pred_label), dim=0)
            if n == 1:
                print(pred_label.argmax(dim=1))

    return data_tmp, pred_test, pred_label_test
