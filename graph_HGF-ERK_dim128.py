
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import glob
import torch_geometric as pyg
import torch_geometric.data as data
import math
import torch_geometric.utils as pyg_utils
import torch.nn as nn
from torch.nn import functional as F
from shutil import copyfile
from tensorboardX import SummaryWriter
from prettytable import PrettyTable
from scipy.spatial import Voronoi, voronoi_plot_2d
from tifffile import imwrite, imread
from scipy.ndimage import gaussian_filter
import torch_geometric.transforms as T
from geomloss import SamplesLoss
import time

def voronoi_finite_polygons_2d(vor, radius=0.05):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region

        try:
            ridges = all_ridges[p1]
        except:
            continue

        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
def explode_xy(xy):
    xl=[]
    yl=[]
    for i in range(len(xy)):
        xl.append(xy[i][0])
        yl.append(xy[i][1])
    return xl,yl
def shoelace_area(x_list,y_list):
    a1,a2=0,0
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    for j in range(len(x_list)-1):
        a1 += x_list[j]*y_list[j+1]
        a2 += y_list[j]*x_list[j+1]
    l=abs(a1-a2)/2
    return l
class OTLoss():

    def __init__(self, device):
        blur = 0.05
        self.ot_solver = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9, debias=True)
        self.device = device

    def __call__(self, x,y):
        loss_xy = self.ot_solver(x,y)
        return loss_xy

class Erk_move(nn.Module):
    def __init__(self, GNN, x, data, data_id, device):    # in_feats=2, out_feats=2, num_layers=3, hidden=128):

        super(Erk_move, self).__init__()

        self.device = device
        self.GNN= GNN
        self.x0 = x
        self.dataset = data
        self.ff = data_id
        # self.old_x = nn.Parameter(x[:, :].clone().detach().requires_grad_(False))
        self.new_x = nn.Parameter(x[:,:].clone().detach().requires_grad_(True))

    def forward(self):

        dataset = data.Data(x=self.new_x.clone().detach(), pos=self.new_x[:,2:4].detach())
        transform = T.Compose([T.Delaunay(),T.FaceToEdge(),T.Distance(norm=False)])
        dataset = transform(dataset)
        distance = dataset.edge_attr.detach().cpu().numpy()
        pos = np.argwhere(distance<0.15)
        edges = dataset.edge_index

        # x_temp=torch.cat((self.old_x[:,0:6],self.new_x,self.old_x[:,8:19]),dim=1)
        dataset = data.Data(x=self.new_x, edge_index=edges[:, pos[:, 0]],edge_attr=torch.tensor(distance[pos[:, 0]], device=self.device))
        pred = self.GNN(data=dataset, data_id=self.ff)

        return pred

class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers, device):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layernorm = False

        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size, device=device, dtype=torch.float64
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
                # self.layers.append(torch.nn.Dropout(p=0.0))
        if self.layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size, device=device, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class EdgeNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, hidden_size, layers):
        super().__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        return self.new_edges

    def message(self, x_i, x_j, edge_feature):

        xi = x_i[:, 0:1]
        yi = x_i[:, 1:2]

        xj = x_j[:, 0:1]
        yj = x_j[:, 1:2]

        vxi = x_i[:, 2:3]
        vyi = x_i[:, 3:4]

        vxj = x_j[:, 2:3]
        vyj = x_j[:, 3:4]

        erki = x_i[:,4:5]
        erkj = x_j[:,4:5]

        dx = xj - xi
        dy = yj - yi
        dvx = vxj - vxi
        dvy = vyj - vyi

        d = edge_feature[:,0:1]

        self.new_edges = torch.cat((d, dx , dy , dvx, dvy), dim=-1)

        return d
class InteractionNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, hidden_size, layers, h, msg, device):
        super().__init__(aggr='add')  # "Add" aggregation.

        self.hidden_size = hidden_size
        self.h = h
        self.layers = layers
        self.msg = msg
        self.device =device

        if (self.msg == 0) | (self.msg==3):
             self.lin_edge = MLP(input_size=1, hidden_size=self.hidden_size, output_size=7, layers=self.layers, device=self.device)
        if (self.msg == 1):
            self.lin_edge = MLP(input_size=13, hidden_size=self.hidden_size, output_size=7, layers=self.layers, device=self.device)
        if (self.msg == 2):
            self.lin_edge = MLP(input_size=14, hidden_size=self.hidden_size, output_size=7, layers=self.layers, device=self.device)

        if self.h == 0:
            self.lin_node = MLP(input_size=12, hidden_size=self.hidden_size, output_size=7, layers=self.layers, device=self.device)
        else:
            self.lin_node = MLP(input_size=64, hidden_size=self.hidden_size, output_size=7, layers=self.layers, device=self.device)

        self.rnn = torch.nn.GRUCell(14, 64,device=self.device,dtype=torch.float64)

    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        if self.h==0:
            node_out = self.lin_node(torch.cat((x[:,:-2], aggr), dim=-1))
            prev_node_feature = node_out[:, :-2]
            prev_edge_feature = edge_feature[:,:-2]

            node_out = x + node_out
            node_out[:, :-2] = prev_node_feature  # no update ui

            edge_out = edge_feature + model.new_edges
            if model_config['remove_update_U']:
                edge_out[:,:-2] = prev_edge_feature     #no update ui

        else:
            super_x=torch.cat((x, coeff, aggr), dim=-1)
            model.h = self.rnn(super_x,model.h)     ########### TO BE CORRECTED ###################
            node_out = self.lin_node(model.h)
            edge_out = edge_feature + self.new_edges

        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):

        xi = x_i[:, 0:1]
        yi = x_i[:, 1:2]

        xj = x_j[:, 0:1]
        yj = x_j[:, 1:2]

        vxi = x_i[:, 2:3]
        vyi = x_i[:, 3:4]

        vxj = x_j[:, 2:3]
        vyj = x_j[:, 3:4]

        erki = x_i[:,4:5]
        erkj = x_j[:,4:5]


        dx = xj - xi
        dy = yj - yi
        dvx = vxj - vxi
        dvy = vyj - vyi


        diff_erk = erkj - erki

        cell_id =x_i[:, 1].detach().cpu().numpy()
        coeff=model.a[cell_id,:]

        if self.msg==0:
            x = diff_erk*0
        elif mself.msg==1:
            x = torch.cat((edge_feature, dx*coeff[:,0:1], dy*coeff[:,0:1], dvx*coeff[:,1:2], dvy*coeff[:,1:2]), dim=-1)
        elif self.msg==2:
            x = torch.cat((edge_feature, dx*coeff[:,0:1], dy*coeff[:,0:1], dvx*coeff[:,1:2], dvy*coeff[:,1:2], diff_erk), dim=-1)
        else: # model_config['msg']==3:
            x = diff_erk*0

        x = self.lin_edge(x)
        self.new_edges = x

        return x
class InteractionNetworkEmb(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, layers, embedding, device, h):
        super().__init__(aggr='add')  # "Add" aggregation.

        self.h = h
        self.layers = layers
        self.device =device
        self.embedding = embedding

        self.lin_edge = MLP(input_size=3*self.embedding, hidden_size=3*self.embedding, output_size=self.embedding, layers= self.layers, device=self.device)

        if self.h == 0:
            self.lin_node = MLP(input_size=2*self.embedding, hidden_size=2*self.embedding, output_size=self.embedding, layers= self.layers, device=self.device)
        else:
            self.lin_node = MLP(input_size=64, hidden_size=self.embedding, output_size=self.embedding, layers= self.layers, device=self.device)
            self.rnn = torch.nn.GRUCell(14, 64,device=self.device,dtype=torch.float64)

    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        if self.h==0:
            node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
            node_out = x + node_out
            edge_out = edge_feature + self.new_edges

        else:
            super_x=torch.cat((x, aggr), dim=-1)
            self.state = self.rnn(super_x,self.state)
            node_out = self.lin_node(self.state)
            edge_out = edge_feature + self.new_edges

        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):

        x = torch.cat((edge_feature, x_i, x_j ), dim=-1)

        x = self.lin_edge(x)
        self.new_edges = x

        return x

class ResNetGNN(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(self,model_config, device):
        super().__init__()

        self.hidden_size = model_config['hidden_size']
        self.embedding = model_config['embedding']
        self.nlayers = model_config['n_mp_layers']
        self.h = model_config['h']
        self.noise_level = model_config['noise_level']
        self.n_tracks = model_config['n_tracks']
        self.msg=model_config['msg']
        self.device=device
        self.output_angle = model_config['output_angle']

        self.edge_init = EdgeNetwork(self.hidden_size, layers=3)

        if self.embedding > 0:
            self.layer = torch.nn.ModuleList([InteractionNetworkEmb(layers=3, embedding=self.embedding, h=self.h, device=self.device) for _ in range(self.nlayers)])
            self.node_out = MLP(input_size=self.embedding, hidden_size=self.hidden_size, output_size=1, layers=3, device=self.device)
        else:
            self.layer = torch.nn.ModuleList([InteractionNetwork(hidden_size=self.hidden_size, layers=3, h=self.h, msg=self.msg, device=self.device) for _ in range(self.nlayers)])
            self.node_out = MLP(input_size=7, hidden_size=self.hidden_size, output_size=1, layers=3, device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((3,int(self.n_tracks+1), 3)), requires_grad=False, device=self.device))
        self.t = nn.Parameter(torch.tensor(np.ones((3,int(self.n_tracks+1), 3)), requires_grad=False, device=self.device))

        self.h_all = torch.zeros((int(self.n_tracks + 1), 64), requires_grad=False, device='cuda:0', dtype=torch.float64)

        if self.embedding>0:
            self.embedding_node = MLP(input_size=5+3, hidden_size=self.embedding, output_size=self.embedding, layers=3, device=self.device)
            self.embedding_edges = MLP(input_size=5, hidden_size=self.embedding, output_size=self.embedding, layers=3, device=self.device)

    def forward(self, data, data_id):

        node_feature = torch.cat((data.x[:,2:4],data.x[:,6:8],data.x[:,4:5]), dim=-1)
        noise = torch.randn((node_feature.shape[0], node_feature.shape[1]),requires_grad=False, device='cuda:0') * self.noise_level
        node_feature= node_feature+noise
        edge_feature = self.edge_init(node_feature, data.edge_index, edge_feature=data.edge_attr)

        if self.embedding > 0:
            cell_embedding = self.a[data_id, data.x[:, 1].detach().cpu().numpy(), :]
            node_feature = self.embedding_node(torch.cat((node_feature, cell_embedding), dim=-1))
            edge_feature = self.embedding_edges(edge_feature)

        for i in range(self.nlayers):
            node_feature, edge_feature = self.layer[i](node_feature, data.edge_index, edge_feature=edge_feature)
        pred = self.node_out(node_feature)

        return pred

def train_model():
    ntry = model_config['ntry']
    bRollout = model_config['bRollout']
    output_angle = model_config['output_angle']

    print('ntry: ', model_config['ntry'])
    if model_config['h'] == 0:
        print('no GRUcell ')
    elif model_config['h'] == 1:
        print('with GRUcell ')
    if (model_config['msg'] == 0) | (model_config['embedding'] > 0):
        print('msg: 0')
    elif model_config['msg'] == 1:
        print('msg: MLP(x_jp, y_jp, vx_p, vy_p, ux, uy)')
    elif model_config['msg'] == 2:
        print('msg: MLP(x_jp, y_jp, vx_p, vy_p, ux, uy, diff erkj)')
    elif model_config['msg'] == 3:
        print('msg: MLP(diff_erk)')
    else:  # model_config['msg']==4:
        print('msg: 0')
    print('embedding: ', model_config['embedding'])
    print('hidden_size: ', model_config['hidden_size'])
    print('n_mp_layers: ', model_config['n_mp_layers'])
    print('noise_level: ', model_config['noise_level'])
    print('bRollout: ', model_config['bRollout'])
    print('rollout_window: ', model_config['rollout_window'])
    print(f'batch size: ', model_config['batch_size'])
    print('remove_update_U: ', model_config['remove_update_U'])
    print('output_angle: ', model_config['output_angle'])

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)
    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

    model = ResNetGNN(model_config=model_config, device=device)
    # state_dict = torch.load(f"./log/try_{ntry}/models/best_model.pt")
    # model.load_state_dict(state_dict['model_state_dict'])

    print('model = ResNetGNN()   predict derivative Erk ')

    if model_config['train_MLPs'] == False:
        print('No MLPs training watch out')
        state_dict = torch.load(f"./log/try_421/models/best_model_new_emb_concatenate.pt")
        model.load_state_dict(state_dict['model_state_dict'])
        for param in model.layer.parameters():
            param.requires_grad = False
        for param in model.node_out.parameters():
            param.requires_grad = False
        for param in model.embedding_node.parameters():
            param.requires_grad = False
        for param in model.embedding_edges.parameters():
            param.requires_grad = False

        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.layer.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.node_out.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.embedding_node.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.embedding_edges.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")

    if model_config['cell_embedding'] == 0:
        model.a.requires_grad = False
        model.t.requires_grad = False
        print('embedding: a false t false')
    elif model_config['cell_embedding'] == 1:
        model.a.requires_grad = True
        model.t.requires_grad = False
        print('embedding: a true t false')
    else:  # model_config['cell_embedding']=2
        model.a.requires_grad = True
        model.t.requires_grad = True
        print('embedding: a true t true')

    optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)  # , weight_decay=5e-3)
    criteria = nn.MSELoss()
    model.train()

    print('Training model = ResNetGNN() ...')

    ff = 0

    print(file_list[ff])

    trackmate = trackmate_list[ff].copy()
    trackmate_true = trackmate_list[ff].copy()

    best_loss = np.inf

    for epoch in range(1000):

        if epoch == 100:
            optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)  # , weight_decay=5e-3)

        mserr_list = []

        trackmate = trackmate_true.copy()

        for frame in range(20, model_config['frame_end'][ff]-20):  # frame_list:

            pos = np.argwhere(trackmate[:, 0] == frame)

            list_all = pos[:, 0].astype(int)
            mask = torch.tensor(np.ones(list_all.shape[0]), device=device)
            for k in range(len(mask)):
                if trackmate[list_all[k] - 1, 1] != trackmate[list_all[k] + 1, 1]:
                    mask[k] = 0
                if (trackmate[list_all[k], 2] < -0.45) | (trackmate[list_all[k], 3] < -0.52) | (
                        trackmate[list_all[k], 3] > 0.55):
                    mask[k] = 0
            mask = mask[:, None]

            x = torch.tensor(trackmate[list_all, 0:19], device=device)
            target = torch.tensor(trackmate_true[list_all + 1, 4:5], device=device)

            dataset = data.Data(x=x, pos=x[:, 2:4])
            transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
            dataset = transform(dataset)
            distance = dataset.edge_attr.detach().cpu().numpy()
            pos = np.argwhere(distance < model_config['radius'])
            edges = dataset.edge_index
            dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                                edge_attr=torch.tensor(distance[pos[:, 0]], device=device))

            optimizer.zero_grad()
            pred = model(data=dataset, data_id=ff)

            loss = criteria((pred[:, :] + x[:, 4:5]) * mask, target * mask) * 3 

            loss.backward()
            optimizer.step()
            mserr_list.append(loss.item())

            trackmate[list_all + 1, 8:9] = np.array(pred.detach().cpu())
            trackmate[list_all + 1, 4:5] = trackmate[list_all, 4:5] + trackmate[list_all + 1, 8:9]

            # pred, target, mu_v_xy, var_v_xy = model(dataset)
            # sum = (mu_v_xy - target[:, 2:4]) ** 2 / (var_v_xy + 1E-7) + torch.log(var_v_xy)
            # sum = sum * mask
            # sum = torch.std(sum, axis=0)
            # sum = torch.sum(sum ** 2)
            # loss = sum / torch.sum(mask) * 1E3

        print(f"Epoch: {epoch} Loss: {np.round(np.mean(mserr_list), 4)}")

        if (np.mean(mserr_list) < best_loss):
            best_loss = np.mean(mserr_list)
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(log_dir, 'models', 'best_model_new_emb_concatenate.pt'))


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device :{device}')

    file_list=["/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/",\
               "/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210108/",\
               "/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210109/"]

    print('Loading trackmate ...')

    trackmate_list=[]
    for ff in range(3):
        trackmate = np.load(f'{file_list[ff]}/trackmate/transformed_spots_try{315+ff}.npy')
        trackmate[-1, 0] = -1
        trackmate_list.append(trackmate)
        if ff==0:
            n_tracks = np.max(trackmate[:, 1]) + 1
            trackmate_true = trackmate.copy()
            nstd = np.load(f'/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/nstd_try315.npy')
            nmean = np.load(f'/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/nmean_try315.npy')
            c = nstd[6] / nstd[2]

    print('Trackmate quality check...')
    time.sleep(0.5)
    for ff in range(3):
        trackmate = trackmate_list[ff]
        for k in tqdm(range(5, trackmate.shape[0] - 1)):
            if trackmate[k-1, 1] == trackmate[k+1, 1]:

                if np.abs(trackmate[k+1, 6] * c - (trackmate[k+1, 2] - trackmate[k, 2])) > 1E-3:
                    print(f'Pb check vx at row {k}')
                if np.abs(trackmate[k+1, 7] * c - (trackmate[k+1, 3] - trackmate[k, 3])) > 1E-3:
                    print(f'Pb check vy at row {k}')

                if np.abs(trackmate[k+1, 15] - (trackmate[k+1, 6] - trackmate[k, 6])) > 1E-3:
                    print(f'Pb check accx at row {k}')
                if np.abs(trackmate[k+1, 16] - (trackmate[k+1, 7] - trackmate[k, 7])) > 1E-3:
                    print(f'Pb check accy at row {k}')
    print('... done')

    # model_config = {'ntry': 421,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'embedding': 128,
    #                 'cell_embedding': 1,
    #                 'train_MLPs': True,
    #                 'output_angle': False,
    #                 'n_mp_layers': 5,
    #                 'hidden_size': 32,
    #                 'noise_level': 0,
    #                 'batch_size': 8,
    #                 'bRollout': True,
    #                 'rollout_window': 2,
    #                 'remove_update_U': True,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    #
    # train_model()
    #
    # model_config = {'ntry': 422,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'embedding': 64,
    #                 'cell_embedding': 1,
    #                 'train_MLPs': True,
    #                 'output_angle': False,
    #                 'n_mp_layers': 5,
    #                 'hidden_size': 32,
    #                 'noise_level': 0,
    #                 'batch_size': 8,
    #                 'bRollout': True,
    #                 'rollout_window': 2,
    #                 'remove_update_U': True,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    #
    # train_model()
    #
    # model_config = {'ntry': 423,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'embedding': 128,
    #                 'cell_embedding': 1,
    #                 'train_MLPs': True,
    #                 'output_angle': False,
    #                 'n_mp_layers': 10,
    #                 'hidden_size': 32,
    #                 'noise_level': 0,
    #                 'batch_size': 8,
    #                 'bRollout': True,
    #                 'rollout_window': 2,
    #                 'remove_update_U': True,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}

    # train_model()

    # model_config = {'ntry': 422,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'embedding': 128,
    #                 'cell_embedding': 1,
    #                 'train_MLPs': True,
    #                 'output_angle': True,
    #                 'n_mp_layers': 5,
    #                 'hidden_size': 32,
    #                 'noise_level': 0,
    #                 'batch_size': 4,
    #                 'bRollout': False,
    #                 'rollout_window': 2,
    #                 'remove_update_U': True,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    #
    # model_config = {'ntry': 423,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'embedding': 128,
    #                 'cell_embedding': 1,
    #                 'train_MLPs': True,
    #                 'output_angle': False,
    #                 'n_mp_layers': 5,
    #                 'hidden_size': 32,
    #                 'noise_level': 0,
    #                 'batch_size': 4,
    #                 'bRollout': True,
    #                 'rollout_window': 2,
    #                 'remove_update_U': True,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}

    # model_config = {'ntry': 424,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'embedding': 128,
    #                 'cell_embedding': 1,
    #                 'train_MLPs': True,
    #                 'output_angle': False,
    #                 'n_mp_layers': 5,
    #                 'hidden_size': 32,
    #                 'noise_level': 0,
    #                 'batch_size': 8,
    #                 'bRollout': True,
    #                 'rollout_window': 2,
    #                 'remove_update_U': True,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    #
    # train_model()

    model_config = {'ntry': 425,
                    'h': 0,
                    'msg': 1,
                    'embedding': 128,
                    'cell_embedding': 0,
                    'train_MLPs': True,
                    'output_angle': False,
                    'n_mp_layers': 5,
                    'hidden_size': 32,
                    'noise_level': 0,
                    'batch_size': 8,
                    'bRollout': True,
                    'rollout_window': 2,
                    'remove_update_U': True,
                    'frame_start': 20,
                    'frame_end': [241, 228, 228],
                    'n_tracks': 3561,
                    'radius': 0.15}

    train_model()





