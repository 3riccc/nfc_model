import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from torch_geometric.nn import GINConv

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Definition of Encoder
class Encoder(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, n_layers):
        super(Encoder,self).__init__()
        
        self.convs = torch.nn.ModuleList()
        
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        for _ in range(n_layers):
            self.convs.append(GINConv(nn.Linear(hidden_dim,hidden_dim)))
#         self.out_proj = torch.nn.Linear((n_layers+1)*hidden_dim+1, output_dim)
        self.out_proj = torch.nn.Linear((n_layers+1)*hidden_dim, output_dim)
    def forward(self,x,edge_index):
        x = self.in_proj(x)
        hidden_states = [x]
        for layer in self.convs:
            x = layer(x,edge_index)
            hidden_states.append(x)
#         hidden_states.append(size.unsqueeze(0).repeat(x.shape[0],1))
        x = torch.cat(hidden_states, dim=1)
        x = self.out_proj(x)
#         x = F.gumbel_softmax(x,dim=-1,hard=True)
        x = F.softmax(x,dim=-1)
        return x

# Definition of Decoder
class WeightSumDecoder(nn.Module):
    def __init__(self,macro_size):
        super(WeightSumDecoder,self).__init__()
        self.scale_mat = torch.ones(macro_size,macro_size)-torch.eye(macro_size)*0.5
        self.scale_mat = self.scale_mat.to(device)
    def forward(self,S,A):
        adj = S.t()@A@S*(S.shape[1]/S.shape[0])
        adj *= self.scale_mat
        return adj