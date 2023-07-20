import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.data import HeteroData
import numpy as np
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as datA
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
from database import get_data


# 定义异质图编辑器
class HeteroEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HeteroEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)

    def forward(self, data):
        if 'middle' in data.node_types:
            a = torch.cat([data["input"].y, torch.zeros_like(data["input"].y)], dim=-1)
            b = torch.cat([data["output"].y, torch.zeros_like(data["output"].y)], dim=-1)
            x = torch.cat([data["input"].x, data["output"].x, a, b], dim=0)
            x = self.fc1(x)
            x = F.elu(x)
        else:
            a = torch.cat([data["input"].y, torch.zeros_like(data["input"].y)], dim=-1)
            b = torch.cat([data["output"].y, torch.zeros_like(data["output"].y)], dim=-1)
            x = torch.cat([data["input"].x, data["output"].x, a, b], dim=0)
            x = self.fc1(x)
            x = F.elu(x)
        return x


class HeteroDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HeteroDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


encoder = HeteroEncoder(input_dim=2, hidden_dim=16)
input_dim = 16  # 输入特征维度
output_dim = 2  # 输出特征维度
decoder = HeteroDecoder(input_dim, output_dim)
criterion = nn.MSELoss()
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)


def train(encoder, decoder, data_list, encoder_optimizer, decoder_optimizer, criterion, num_epochs):
    encoder.train()
    decoder.train()

    for epoch in range(num_epochs):
        count = 0
        for data in data_list:
            count += 1
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            a = torch.cat([data["input"].y, torch.zeros_like(data["input"].y)], dim=-1)
            b = torch.cat([data["output"].y, torch.zeros_like(data["output"].y)], dim=-1)
            q = torch.cat([data["input"].x, data["output"].x, a, b], dim=0)
            x1 = encoder(data)
            d = decoder(x1)
            loss = criterion(q, d)

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

data_list = get_data()
num_epochs = 10
train(encoder, decoder, data_list, encoder_optimizer, decoder_optimizer, criterion, num_epochs)

# 输出节点嵌入
for data in data_list:
    encoder.eval()
    x = encoder(data)
    print('编码器的输出为',x)
    x1 = decoder(x)
    print('解码器的输出为',x1)
print('第san版本')