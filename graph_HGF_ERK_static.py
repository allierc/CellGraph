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
class OTLoss():

    def __init__(self, device):
        blur = 0.05
        self.ot_solver = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9, debias=True)
        self.device = device

    def __call__(self, x, y):
        loss_xy = self.ot_solver(x, y)
        return loss_xy




class Erk_move(nn.Module):
    def __init__(self, GNN, x, data, data_id, device):    # in_feats=2, out_feats=2, num_layers=3, hidden=128):

        super(Erk_move, self).__init__()

        self.device = device
        self.GNN= GNN
        self.x0 = x
        self.dataset = data
        self.ff = data_id
        self.new_x = nn.Parameter(x.clone().detach().requires_grad_(True))

    def forward(self):

        dataset = data.Data(x=self.new_x.clone().detach(), pos=self.new_x[:, 2:4].detach())
        transform = T.Compose([T.Delaunay(),T.FaceToEdge(),T.Distance(norm=False)])
        dataset = transform(dataset)
        distance = dataset.edge_attr.detach().cpu().numpy()
        pos = np.argwhere(distance<0.15)
        edges = dataset.edge_index
        dataset = data.Data(x=self.new_x, edge_index=edges[:, pos[:, 0]],edge_attr=torch.tensor(distance[pos[:, 0]], device=self.device))
        message, pred = self.GNN(data=dataset, data_id=self.ff)

        return pred

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, device):    # in_feats=2, out_feats=2, num_layers=3, hidden=128):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        layer = nn.Linear(input_size, hidden_size, device=device, dtype=torch.float64)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(layer)
        if layers > 2:
            for i in range(1, layers - 1):
                layer = nn.Linear(hidden_size, hidden_size, device=device, dtype=torch.float64)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden_size, output_size, device=device, dtype=torch.float64)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(layer)

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x
class InteractionParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):   #  in_feats=9, out_feats=2, num_layers=2, hidden=16):

        self.device=device
        self.h = model_config['h']
        self.msg=model_config['msg']
        self.aggr=model_config['aggr']
        self.cell_embedding = model_config['cell_embedding']
        self.embedding = model_config['embedding']
        self.nlayers = model_config['n_mp_layers']
        self.hidden_size = model_config['hidden_size']
        self.noise_level = model_config['noise_level']
        self.n_tracks = model_config['n_tracks']
        self.rot_mode = model_config['rot_mode']

        if (self.aggr == 0):
            super(InteractionParticles, self).__init__(aggr='add')  # "Add" aggregation.
        else:  # mode_aggr==1
            super(InteractionParticles, self).__init__(aggr='mean')  # "Mean" aggregation.

        if (self.msg == 0) | (self.msg  == 4):
            self.lin_edge = MLP(input_size=1 + self.embedding, hidden_size=self.hidden_size, output_size=1, layers=self.nlayers, device=self.device)
        if (self.msg == 1):
            self.lin_edge = MLP(input_size=4 + self.embedding, hidden_size=self.hidden_size, output_size=1, layers=self.nlayers, device=self.device)
        if (self.msg == 2) | (self.msg == 3) | (self.msg == 5):
            self.lin_edge = MLP(input_size=2 + self.embedding, hidden_size=self.hidden_size, output_size=1, layers=self.nlayers, device=self.device)

        if self.h  == 0:
            self.lin_update = MLP(input_size=1, hidden_size=16, output_size=1, layers=2, device=self.device)
        if self.h  == 1:
            self.lin_update = MLP(input_size=3, hidden_size=16, output_size=1, layers=2, device=self.device)
        if self.h  == 2:
            self.lin_update = MLP(input_size=2, hidden_size=16, output_size=1, layers=2, device=self.device)
        if self.h  > 2:
            self.lin_update = MLP(input_size=1, hidden_size=16, output_size=1, layers=2, device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((3,int(self.n_tracks+1), 3)), requires_grad=False, device=self.device))
        self.t = nn.Parameter(torch.tensor(np.ones((3,int(self.n_tracks+1), 3)), requires_grad=False, device=self.device))

    def forward(self, data, data_id):

        self.data_id=data_id

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        noise = torch.randn((x.shape[0], x.shape[1]),requires_grad=False, device=self.device ) * self.noise_level
        x=x+noise

        phi = torch.randn([x.shape[0]], dtype=torch.float32, requires_grad=False, device=self.device) * np.pi * 2
        cos = torch.cos(phi)
        sin = torch.sin(phi)

        x = torch.cat((x, cos[:, None], sin[:, None]), axis=-1)
        message = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        if self.h<2:
            if self.h  == 0:
                pred = self.lin_update(message)
            elif self.h  == 1:
                area_i = x[:, 11:12]
                perimeter_i = x[:, 12:13]
                pred = self.lin_update(torch.cat((area_i, perimeter_i, message), dim=-1))

        coeff = self.a[self.data_id,x[:, 1].detach().cpu().numpy()]
        if self.h==2:
            vxi_p = x[:, 6:7] * x[:, 14:15] + x[:, 7:8] * x[:, 15:16]
            vyi_p = -x[:, 6:7] * x[:, 15:16] + x[:, 7:8] * x[:, 14:15]
            pred = self.lin_update(torch.cat((vxi_p* coeff[:, 0:1], vyi_p* coeff[:, 0:1]), dim=-1))
        if self.h==3:
            area_i = x[:, 11:12]
            pred = self.lin_update(area_i* coeff[:, 0:1] )
        if self.h==4:
            nucleus_i = x[:, 5:6]
            pred = self.lin_update(nucleus_i* coeff[:, 0:1])
        if self.h==5:
            cell_conc_i = x[:, 13:14]
            pred = self.lin_update(cell_conc_i* coeff[:, 0:1])

        return message, pred

    def message(self, x_i, x_j, edge_attr, edge_index):

        cos_i = x_i[:, -2:-1]
        sin_i = x_i[:, -1:]
        cos_j = x_j[:, -2:-1]
        sin_j = x_j[:, -1:]

        xi = x_i[:, 2:3]
        yi = x_i[:, 3:4]
        xj = x_j[:, 2:3]
        yj = x_j[:, 3:4]
        vxi = x_i[:, 6:7]
        vyi = x_i[:, 7:8]
        vxj = x_j[:, 6:7]
        vyj = x_j[:, 7:8]

        erkj = x_j[:, 4:5]
        nucleus_j = x_j[:, 5:6]
        deg_j = x_j[:, 10:11]
        area_j = x_j[:, 11:12]
        perimeter_j = x_j[:, 12:13]

        if self.rot_mode == 0:
            x_jp = xj - xi
            y_jp = yj - yi
            vx_p = vxj
            vy_p = vyj

        if self.rot_mode == 1:
            x_jp = (xj - xi) * cos_i + (yj - yi) * sin_i
            y_jp = -(xj - xi) * sin_i + (yj - yi) * cos_i
            vx_p = (vxj - vxi) * cos_i + (vyj - vyi) * sin_i
            vy_p = -(vxj - vxi) * sin_i + (vyj - vyi) * cos_i

        if self.rot_mode == 2:
            x_jp = (xj - xi) * cos_i + (yj - yi) * sin_i
            y_jp = -(xj - xi) * sin_i + (yj - yi) * cos_i
            vx_p = (vxj - vxi) * cos_j + (vyj - vyi) * sin_j
            vy_p = -(vxj - vxi) * sin_j + (vyj - vyi) * cos_j

        d = torch.clamp(edge_attr, 0.01)

        cell_id = x_i[:, 1].detach().cpu().numpy()
        coeff = self.a[self.data_id,cell_id]


        if self.embedding == 0:
            if self.msg == 0:
                return erkj * 0
            elif self.msg == 1:
                in_features = torch.cat((x_jp * coeff[:, 0:1], y_jp * coeff[:, 0:1], vx_p / d * coeff[:, 1:2], vy_p / d * coeff[:, 1:2]),dim=-1)
            elif self.msg == 2:
                in_features = torch.cat((vx_p / d * coeff[:, 1:2], vy_p / d * coeff[:, 1:2]), dim=-1)
            elif self.msg == 3:
                in_features = torch.cat((x_jp * coeff[:, 0:1], y_jp * coeff[:, 0:1]), dim=-1)
            elif self.msg == 4:
                return erkj
            else:  # mode_m=5:
                in_features = torch.cat((area_j * coeff[:, 0:1], perimeter_j * coeff[:, 1:2]), dim=-1)
            out = self.lin_edge(in_features)

        else:
            if self.msg == 0:
                return erkj * 0
            elif self.msg == 1:
                in_features = torch.cat((x_jp , y_jp , vx_p / d , vy_p / d ),dim=-1)
            elif self.msg == 2:
                in_features = torch.cat((vx_p / d , vy_p / d ), dim=-1)
            elif self.msg == 3:
                in_features = torch.cat((x_jp , y_jp ), dim=-1)
            elif self.msg == 4:
                return erkj
            else:  # mode_m=5:
                in_features = torch.cat((area_j * coeff[:, 0:1], perimeter_j * coeff[:, 1:2]), dim=-1)
            out =  self.lin_edge(torch.cat((in_features,coeff), dim=-1))

        return out





def train_model():
    trackmate_true = trackmate_list[0]

    dloss = OTLoss(device)

    ntry = model_config['ntry']
    bRollout = model_config['bRollout']
    bNoise = model_config['bNoise']

    print(f'Training {ntry}')
    print(f'rot_mode:', model_config['rot_mode'])
    if model_config['h'] == 0:
        print('H: MLP(message)')
    elif model_config['h'] == 1:
        print('H: MLP(area perimeter message)')
    elif model_config['h'] == 2:
        print('H: MLP(vx_p, vy_p)')
    elif model_config['h'] == 3:
        print('H: MLP(area)')
    elif model_config['h'] == 4:
        print('H: MLP(nucleus area)')
    else: # model_config['h'] == 5:
        print('H: MLP(cell concentration)')


    if model_config['msg'] == 0:
        print('msg: 0')
    elif model_config['msg'] == 1:
        print('msg: MLP(x_jp, y_jp, vx_p/d, vy_p/d)')
    elif model_config['msg'] == 2:
        print('msg: MLP(vx_p/d, vy_p/d)')
    elif model_config['msg'] == 3:
        print('msg: MLP(x_jp, y_jp)')
    elif model_config['msg'] == 4:
        print('msg: MLP(erkj)')
    else:  # mode_m=5:
        print('msg: MLP(area, perimeter)')

    if model_config['aggr'] == 0:
        print('aggregation add')
    else:  # mode_aggr==1:
        print('aggregation mean')

    if model_config['cell_embedding'] == 0:
        print('embedding: a false ')
    if model_config['cell_embedding'] == 1:
        print('embedding: a true ')
    if model_config['cell_embedding'] == 2:
        print('embedding: a true concatenate')

    print('Bnoise:', model_config['bNoise'])
    print('noise_level:', model_config['noise_level'])
    print(' ')

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)
    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

    model = InteractionParticles(model_config=model_config, device=device)
    # state_dict = torch.load(f"./log/try_{ntry}/models/best_model.pt")
    # model.load_state_dict(state_dict['model_state_dict'])

    criteria = nn.MSELoss()
    model.train()
    best_loss = np.inf

    if model_config['cell_embedding'] == 0:
        model.a.requires_grad = False
    elif model_config['cell_embedding'] > 0:
        model.a.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)  # , weight_decay=5e-3)

    print('Training model = InteractionParticles() ...')

    for epoch in range(500):

        if epoch == 400:
            optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)  # , weight_decay=5e-3)

        mserr_list = []
        datafit_list = []
        distributionfit_list = []

        ff = 0
        trackmate = trackmate_list[ff].copy()
        trackmate_true = trackmate_list[ff].copy()

        f = np.random.permutation(np.arange(20, model_config['frame_end'][ff] - 5 - 1))

        for frame in f:  # frame_list:

            pos = np.argwhere(trackmate[:, 0] == frame)

            list_all = pos[:, 0].astype(int)
            mask = torch.tensor(np.ones(list_all.shape[0]), device=device)
            for k in range(len(mask)):
                if trackmate[list_all[k] - 1, 1] != trackmate[list_all[k] + 1, 1]:
                    mask[k] = 0
            mask = mask[:, None]
            x = torch.tensor(trackmate[list_all, 0:14], device=device)
            dataset = data.Data(x=x, pos=x[:, 2:4])
            transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
            dataset = transform(dataset)
            distance = dataset.edge_attr.detach().cpu().numpy()
            pos = np.argwhere(distance < model_config['radius'])
            edges = dataset.edge_index
            dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                                edge_attr=torch.tensor(distance[pos[:, 0]], device=device))

            optimizer.zero_grad()

            target = x[:, 4:5]
            message, pred = model(data=dataset, data_id=ff)

            if False:  # frame%10 == 0:
                distributionfit = dloss(pred * mask, target * mask) * 1E2
                distributionfit_list.append(distributionfit.item())
            else:
                distributionfit = 0

            datafit = criteria(pred * mask, target * mask) * 3
            loss = datafit + distributionfit

            loss.backward()
            optimizer.step()

            mserr_list.append(loss.item())
            datafit_list.append(datafit.item())

        print(f"Epoch {epoch} Loss: {np.round(np.mean(mserr_list), 4)}")

        if (np.mean(mserr_list) < best_loss):
            best_loss = np.mean(mserr_list)
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},os.path.join(log_dir, 'models', 'best_model_new_emb_concatenate.pt'))

    print(' end of training')
    print(' ' )


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Device :{device}')

    file_list=["/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/",\
               "/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210108/",\
               "/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210109/"]
    file_folder = file_list[0]
    print(file_folder)

    print('Loading trackmate ...')

    trackmate_list=[]
    for ff in range(1):
        trackmate = np.load(f'{file_list[ff]}/trackmate/transformed_spots_try{415+ff}.npy')
        trackmate[-1, 0] = -1
        trackmate_list.append(trackmate)
        if ff==0:
            n_tracks = np.max(trackmate[:, 1]) + 1
            trackmate_true = trackmate.copy()
            nstd = np.load(f'{file_folder}/trackmate/nstd_try415.npy')
            nmean = np.load(f'{file_folder}/trackmate/nmean_try415.npy')
            c = nstd[6] / nstd[2]

    print('Trackmate quality check...')
    time.sleep(0.5)
    for ff in range(1):
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

    model_config = {'ntry': 440,
                    'h': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode':1,
                    'embedding': 3,
                    'cell_embedding': 1,
                    'time_embedding': False,
                    'n_mp_layers': 3,
                    'hidden_size': 32,
                    'bNoise': False,
                    'noise_level': 0,
                    'batch_size': 4,
                    'bRollout': False,
                    'rollout_window': 2,
                    'frame_start': 20,
                    'frame_end': [241, 228, 228],
                    'n_tracks': 3561,
                    'radius': 0.15}
    train_model()

    model_config = {'ntry': 441,
                    'h': 0,
                    'msg': 2,
                    'aggr': 0,
                    'rot_mode':1,
                    'embedding': 3,
                    'cell_embedding': 1,
                    'time_embedding': False,
                    'n_mp_layers': 3,
                    'hidden_size': 32,
                    'bNoise': False,
                    'noise_level': 0,
                    'batch_size': 4,
                    'bRollout': False,
                    'rollout_window': 2,
                    'frame_start': 20,
                    'frame_end': [241, 228, 228],
                    'n_tracks': 3561,
                    'radius': 0.15}
    train_model()

    model_config = {'ntry': 442,
                    'h': 0,
                    'msg': 3,
                    'aggr': 0,
                    'rot_mode':1,
                    'embedding': 3,
                    'cell_embedding': 1,
                    'time_embedding': False,
                    'n_mp_layers': 3,
                    'hidden_size': 32,
                    'bNoise': False,
                    'noise_level': 0,
                    'batch_size': 4,
                    'bRollout': False,
                    'rollout_window': 2,
                    'frame_start': 20,
                    'frame_end': [241, 228, 228],
                    'n_tracks': 3561,
                    'radius': 0.15}
    train_model()

    # model_config = {'ntry': 343,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 0,
    #                 'cell_embedding': 1,
    #                 'time_embedding': False,
    #                 'n_mp_layers': 3,
    #                 'hidden_size': 32,
    #                 'bNoise': False,
    #                 'noise_level': 0,
    #                 'batch_size': 4,
    #                 'bRollout': False,
    #                 'rollout_window': 2,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    # train_model()

    # model_config = {'ntry': 344,
    #                 'h': 0,
    #                 'msg': 2,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 0,
    #                 'cell_embedding': 1,
    #                 'time_embedding': False,
    #                 'n_mp_layers': 3,
    #                 'hidden_size': 32,
    #                 'bNoise': False,
    #                 'noise_level': 0,
    #                 'batch_size': 4,
    #                 'bRollout': False,
    #                 'rollout_window': 2,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    # train_model()
    #
    # model_config = {'ntry': 345,
    #                 'h': 0,
    #                 'msg': 3,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 0,
    #                 'cell_embedding': 1,
    #                 'time_embedding': False,
    #                 'n_mp_layers': 3,
    #                 'hidden_size': 32,
    #                 'bNoise': False,
    #                 'noise_level': 0,
    #                 'batch_size': 4,
    #                 'bRollout': False,
    #                 'rollout_window': 2,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    # train_model()
    #
    #
    # model_config = {'ntry': 346,
    #                 'h': 1,
    #                 'msg': 5,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 0,
    #                 'cell_embedding': 1,
    #                 'time_embedding': False,
    #                 'n_mp_layers': 3,
    #                 'hidden_size': 32,
    #                 'bNoise': False,
    #                 'noise_level': 0,
    #                 'batch_size': 4,
    #                 'bRollout': False,
    #                 'rollout_window': 2,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    # train_model()
    #
    #
    # model_config = {'ntry': 347,
    #                 'h': 2,
    #                 'msg': 0,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 0,
    #                 'cell_embedding': 1,
    #                 'time_embedding': False,
    #                 'n_mp_layers': 3,
    #                 'hidden_size': 32,
    #                 'bNoise': False,
    #                 'noise_level': 0,
    #                 'batch_size': 4,
    #                 'bRollout': False,
    #                 'rollout_window': 2,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    # train_model()
    #
    # model_config = {'ntry': 348,
    #                 'h': 3,
    #                 'msg': 0,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 0,
    #                 'cell_embedding': 1,
    #                 'time_embedding': False,
    #                 'n_mp_layers': 3,
    #                 'hidden_size': 32,
    #                 'bNoise': False,
    #                 'noise_level': 0,
    #                 'batch_size': 4,
    #                 'bRollout': False,
    #                 'rollout_window': 2,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    # train_model()
    #
    # model_config = {'ntry': 349,
    #                 'h': 4,
    #                 'msg': 0,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 0,
    #                 'cell_embedding': 1,
    #                 'time_embedding': False,
    #                 'n_mp_layers': 3,
    #                 'hidden_size': 32,
    #                 'bNoise': False,
    #                 'noise_level': 0,
    #                 'batch_size': 4,
    #                 'bRollout': False,
    #                 'rollout_window': 2,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    # train_model()
    #
    # model_config = {'ntry': 350,
    #                 'h': 5,
    #                 'msg': 0,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 0,
    #                 'cell_embedding': 1,
    #                 'time_embedding': False,
    #                 'n_mp_layers': 3,
    #                 'hidden_size': 32,
    #                 'bNoise': False,
    #                 'noise_level': 0,
    #                 'batch_size': 4,
    #                 'bRollout': False,
    #                 'rollout_window': 2,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    # train_model()


    # training_list=[[340,0,1,0,0,0],[341,0,2,0,0,0],[342,0,3,0,0,0],[343,0,4,0,0,0], [344,1,5,0,0,0], [345,0,1,1,0,0],[346,0,2,1,0,0] ]        # rot_mode = 0   cell embedding
    # training_list = [[350, 0, 1, 0, 0, 1], [351, 0, 2, 0, 0, 1], [352, 0, 3, 0, 0, 1], [353, 0, 4, 0, 0, 1], [354, 1, 5, 0, 0, 1], [355, 0, 1, 1, 0, 1], [346, 0, 2, 1, 0, 1]]    # rot_mode = 0  no cell embedding
    # training_list = [[360, 0, 1, 0, 1, 0], [361, 0, 2, 0, 1, 0], [362, 0, 3, 0, 1, 0], [363, 0, 4, 0, 1, 0],
    #  training_list = [[364, 1, 5, 0, 1, 0], [365, 0, 1, 1, 1, 0], [366, 0, 2, 1, 1, 0]]]  # rot_mode = 1   cell embeddin
    # training_list = [[360, 0, 1, 0, 1, 0]]  # rot_mode = 1   cell embedding  noise introduction

