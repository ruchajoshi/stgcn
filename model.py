import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class transform_before(nn.Module):
    def __init__(self, c_in, num_graphs, before):
        super(transform_before, self).__init__()
        self.c_in = c_in
        self.num_graphs = num_graphs
        self.before = before

    def forward(self, x):
        # print('x.shape', x.shape)
        if self.before == 'temp':
            return x.view(-1, self.c_in, self.num_graphs)
        return x.view(-1, self.c_in)



class temporal_conv_layer(nn.Module):
    def __init__(self, num_graphs, K_t, c_in, c_out, act='relu'):
        super(temporal_conv_layer, self).__init__()
        self.K_t = K_t
        self.c_in = c_in
        self.c_out = c_out
        self.act = act
        self.num_graphs = num_graphs
        self.transform_tensor = transform_before(c_in, num_graphs, 'temp')
        if self.act == 'GLU':
            self.conv = nn.Conv1d(c_in, c_out*2, K_t) 
        else:
            self.conv = nn.Conv1d(c_in, c_out, K_t)

    def forward(self, x):
        x = self.transform_tensor(x)
        self.num_graphs = self.num_graphs - self.K_t + 1
        if self.act == 'GLU':
            return self.num_graphs, F.glu(self.conv(x), 1)
        elif self.act == 'sigmoid':
            return self.num_graphs, torch.sigmoid(self.conv(x))
        return self.num_graphs, torch.relu(self.conv(x))

class spatial_conv_layer(nn.Module):
    def __init__(self, num_graphs, K_s, c_in, c_out):
        super(spatial_conv_layer, self).__init__()
        self.num_graphs = num_graphs
        self.K_s = K_s
        self.c_in = c_in
        self.c_out = c_out
        self.transform_tensor = transform_before(c_in, num_graphs, 'graph')
        self.conv = GCNConv(in_channels=c_in, out_channels=c_out)

    def forward(self, x, edge_index):
        x = self.transform_tensor(x)
        return torch.relu(self.conv(x, edge_index))

class spatio_temporal_block(nn.Module):
    def __init__(self, K_t, K_s, c, num_nodes, num_edges, num_graphs):
        super(spatio_temporal_block, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_graphs = num_graphs # 12
        self.temp_conv1 = temporal_conv_layer(num_graphs=self.num_graphs, K_t=K_t, c_in=c[0], c_out=c[1], act='GLU')
        self.num_graphs = self.num_graphs - K_t + 1 # 10
        self.sp_conv = spatial_conv_layer(num_graphs=self.num_graphs, K_s=1, c_in=c[1], c_out=c[2])
        self.temp_conv2 = temporal_conv_layer(num_graphs=self.num_graphs, K_t=3, c_in=c[2], c_out=c[3], act='GLU')
        self.num_graphs = self.num_graphs - K_t + 1
        self.layer_norm = nn.LayerNorm([c[3], self.num_graphs])

    def forward(self, x):
        self.num_graphs, out = self.temp_conv1(x.x)
        out = self.sp_conv(out, x.edge_index[:, :self.num_graphs*self.num_edges])
        self.num_graphs, out = self.temp_conv2(out)
        out = self.layer_norm(out)

        return self.num_graphs, out

class fully_connected_layed(nn.Module):
    def __init__(self, c_in, c_out):
        super(fully_connected_layed, self).__init__()
        self.lin = nn.Linear(c_in, c_out)

    def forward(self, x):
        return self.lin(x)

class output_layer(nn.Module):
    def __init__(self, num_graphs, K_t, c_in, c_out):
        super(output_layer, self).__init__()
        self.num_graphs = num_graphs
        self.temp = temporal_conv_layer(num_graphs=self.num_graphs, K_t=K_t, c_in=c_in, c_out=c_out, act='GLU')
        self.num_graphs = self.num_graphs - 2 * K_t + 2
        self.lin = nn.Linear(2, 1)
        self.fc = fully_connected_layed(c_out, 1)
    
    def forward(self, x):
        self.num_graphs, out = self.temp(x)
        out = self.lin(out)
        self.num_graphs = out.shape[2]
        out = out.reshape(-1, 1, 64)
        out = self.fc(out)
        return self.num_graphs, out

class STGCN(nn.Module):
    def __init__(self, K_t, K_s, c, num_nodes, num_edges, num_graphs):
        super(STGCN, self).__init__()
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.stblock1 = spatio_temporal_block(K_t, K_s, c, num_nodes, num_edges, num_graphs)
        self.num_graphs = self.num_graphs - 2 * K_t + 2
        self.stblock2 = spatio_temporal_block(K_t, K_s, [64, c[1], c[2], c[3]], num_nodes, num_edges, self.num_graphs)
        self.num_graphs = self.num_graphs - 2 * K_t + 2
        self.output_layer = output_layer(num_graphs=self.num_graphs, K_t=K_t, c_in=c[3], c_out=c[3])

    def forward(self, x):
        self.num_graphs, out = self.stblock1(x)
        out = out.permute([0, 2, 1]).reshape([-1, out.shape[1]])
        temp = Data(x=out, edge_index=x.edge_index[:, :self.num_graphs*self.num_edges])
        # print(temp)
        self.num_graphs, out = self.stblock2(temp)
        self.num_graphs, out = self.output_layer(out)
        out = out.reshape(-1, 1)
        return out