import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, classification_report, f1_score

class autoencoder(torch.nn.Module):
    def __init__(self, in_channels, middle_channels, n_layers):
        super().__init__()
        self.n_layers = n_layers + 1
        self.channels = [int((i/(self.n_layers-1))*in_channels+
                             ((self.n_layers-i-1)/(self.n_layers-1))*middle_channels)
                         for i in reversed(range(self.n_layers))]
        self.encode_layers = []
        self.decode_layers = []
        for i in range(self.n_layers-1):
            self.encode_layers.append(Linear(self.channels[i], self.channels[i+1]))
            self.decode_layers.append(Linear(self.channels[n_layers-i], self.channels[n_layers-i-1]))
        self.encode_layer = ModuleList(self.encode_layers)
        self.decode_layer = ModuleList(self.decode_layers)

    def forward(self, input, dropout):
        x = input
        for i in range(self.n_layers-1):
            x = self.encode_layer[i](x)
            x = x.relu()
            x = F.dropout(x, p=dropout, training=self.training)
        out_1 = x
        for i in range(self.n_layers-1):
            x = self.decode_layer[i](x)
            if i != self.n_layers-2:
                x = x.relu()
                x = F.dropout(x, p=dropout, training=self.training)
            else:
                # x = x.sigmoid()
                x = x.relu()
        out_2 = x
        return out_1, out_2

def train_autoencoder(model, data_loader, dropout, criterion, optimizar, device):
    model.train()
    model = model.to(device)

    training_loss = []

    for data in data_loader:
        data = data.to(device)
        reduce_feat, out = model(data, dropout)
        loss = criterion(out, data)
        loss.backward()
        optimizar.step()
        optimizar.zero_grad()

        running_loss = loss.item()
        training_loss.append(running_loss)

def eval_autoencoder(model, data_loader, dropout, criterion, device):
    model.eval()
    model = model.to(device)

    testing_loss = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            reduce_feat, out = model(data, dropout)
            loss = criterion(out, data)

            running_loss = loss.item()
            testing_loss.append(running_loss)

    return testing_loss

def test_autoencoder(model, data_loader, dropout, device, data_tmp):
    model.eval()
    model = model.to(device)

    data_tmp = data_tmp.to(device)

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            reduce_feat, out = model(data, dropout)
            name_tmp = torch.zeros(1, 2, dtype=torch.float).to(device)
            reduce_feat = torch.cat((name_tmp, reduce_feat), dim=1)
            data_tmp = torch.cat((data_tmp, reduce_feat), dim=0)

    return data_tmp