
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

class InteractionParticlesRollout(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, model_config, device):   # in_feats=9, out_feats=2, num_layers=2, hidden=16):

        super(InteractionParticlesRollout, self).__init__(aggr='add')  # "Add" aggregation.

        self.device=device
        self.h = model_config['h']
        self.msg=model_config['msg']
        self.aggr=model_config['aggr']
        self.embedding = model_config['embedding']
        self.cell_embedding = model_config['cell_embedding']
        self.nlayers = model_config['n_mp_layers']
        self.hidden_size = model_config['hidden_size']
        self.noise_level = model_config['noise_level']
        self.n_tracks = model_config['n_tracks']
        self.rot_mode = model_config['rot_mode']
        self.frame_end = model_config['frame_end']

        if (self.msg == 0) | (self.msg == 4) :
            self.lin_edge = MLP(input_size=1 + self.embedding, hidden_size=self.hidden_size, output_size=1, layers=self.nlayers, device=self.device)
        elif (self.msg == 2) :
            self.lin_edge = MLP(input_size=4 + self.embedding, hidden_size=self.hidden_size, output_size=1, layers=self.nlayers, device=self.device)
        elif (self.msg == 3):
            self.lin_edge = MLP(input_size=4 + self.embedding, hidden_size=self.hidden_size, output_size=1, layers=self.nlayers, device=self.device)
        else: #(self.msg==1):
            self.lin_edge = MLP(input_size=5 + self.embedding, hidden_size=self.hidden_size, output_size=1, layers=self.nlayers, device=self.device)

        self.lin_update = MLP(input_size=1, hidden_size=16, output_size=1, layers=2, device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((3,int(self.n_tracks+1), self.embedding)), requires_grad=False, device=self.device))
        self.t = nn.Parameter(torch.tensor(np.ones((3,241, 1)), requires_grad=True, device=self.device))
        # self.a = nn.Parameter(torch.tensor(np.ones((int(self.n_tracks), 3)), requires_grad=False, device=self.device))
        # self.t = nn.Parameter(torch.tensor(np.ones((241, 1)), device='cuda:0', requires_grad=True))

        self.b = nn.Parameter(torch.tensor(np.ones((3,int(self.n_tracks+1),3)), device=self.device, requires_grad=True))

    def forward(self, data, data_id):

        self.data_id = data_id

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        phi = torch.randn([x.shape[0]], dtype=torch.float32, requires_grad=False, device=self.device) * np.pi * 2
        cos = torch.cos(phi)
        sin = torch.sin(phi)

        x = torch.cat((x, cos[:, None], sin[:, None]), axis=-1)

        message = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        # derivative of ERK activity
        cell_id = x[:, 1].detach().cpu().numpy()
        coeff = self.b[self.data_id, cell_id]

        if self.h == 0 :
            pred = self.lin_update(message[:,0:1]) - coeff[:,0:1]*(x[:,8:9]-coeff[:,1:2])
        elif self.h==1:
            pred = self.lin_update(message[:,0:1])

        return message, pred

    def message(self, x_i, x_j,edge_attr):

        cos_i = x_i[:, -2:-1]
        sin_i = x_i[:, -1:]
        cos_j = x_j[:, -2:-1]
        sin_j = x_j[:, -1:]

        xi=x_i[:, 2:3]
        yi=x_i[:, 3:4]
        xj=x_j[:, 2:3]
        yj=x_j[:, 3:4]
        vxi=x_i[:, 4:5]
        vyi=x_i[:, 5:6]
        vxj=x_j[:, 4:5]
        vyj=x_j[:, 5:6]

        diff_erk = x_j[:, 8:9] / torch.clamp(torch.sqrt((x_i[:, 16:17] * self.nstd[16] + self.nmean[16]) * (x_j[:, 16:17] * self.nstd[16] + self.nmean[16])), min=1)

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

        cell_id=x_i[:, 1].detach().cpu().numpy()
        coeff = self.a[self.data_id,cell_id] * self.t[self.data_id,self.frame]
        #coeff = self.a[cell_id] * self.t[self.frame]

        d = torch.clamp(edge_attr,min=0.01)

        if self.embedding == 0:

            if self.msg==0:
                return diff_erk*0
            elif self.msg==1:
                in_features = torch.cat((x_jp * coeff[:, 0:1], y_jp * coeff[:, 0:1], vx_p/d * coeff[:, 1:2], vy_p/d * coeff[:, 1:2], diff_erk * coeff[:, 2:3]), dim=-1)
            elif self.msg==2:
                in_features = torch.cat((x_jp * coeff[:, 0:1], y_jp * coeff[:, 0:1], vx_p/d* coeff[:, 1:2], vy_p/d* coeff[:, 1:2]), dim=-1)
            elif self.msg==3:
                in_features = torch.cat((x_jp * coeff[:, 0:1], y_jp * coeff[:, 0:1], vx_p* coeff[:, 1:2], vy_p* coeff[:, 1:2]), dim=-1)
            elif self.msg==4:
                in_features = diff_erk * coeff[:, 2:3]
            else: # self.msg==4:
                in_features = diff_erk*0
            out = self.lin_edge(in_features)

        else:
            if self.msg==0:
                return diff_erk*0
            elif self.msg==1:
                in_features = torch.cat((x_jp, y_jp, vx_p/d, vy_p/d, diff_erk), dim=-1)
            elif self.msg==2:
                in_features = torch.cat((x_jp, y_jp, vx_p/d, vy_p/d), dim=-1)
            elif self.msg==3:
                in_features = torch.cat((x_jp, y_jp, vx_p, vy_p), dim=-1)
            elif self.msg==4:
                in_features = diff_erk * coeff[:, 2:3]
            else: # self.msg==4:
                in_features = diff_erk*0

            out = self.lin_edge(torch.cat((in_features, coeff), dim=-1))

        return out


    def update(self, aggr_out):

        return aggr_out     #self.lin_node(aggr_out)

def train_model(model_config=None, trackmate=None):

    ntry = model_config['ntry']
    bRollout = model_config['bRollout']

    print(f'Training {ntry}')
    print(f'rot_mode:', model_config['rot_mode'])

    if model_config['h'] == 0:
        print('H: MLP(message - b Erk)')
    else:  # model_config['h'] == 1:
        print('H: MLP(message)')

    if model_config['msg'] == 0:
        print('msg: 0')
    elif model_config['msg'] == 1:
        print('msg: MLP(x_jp, y_jp, vx_p/d, vy_p/d, diff_erk)')
    elif model_config['msg'] == 2:
        print('msg: MLP(x_jp, y_jp, vx_p/d, vy_p/d)')
    elif model_config['msg'] == 3:
        print('msg: MLP(x_jp, y_jp, vx_p, vy_p)')
    elif model_config['msg'] == 4:
        print('msg: MLP(diff_erk)')
    else:  # model_config['msg']==4:
        print('msg: 0')
    if model_config['cell_embedding'] == 0:
        print('embedding: a false t false')
    elif model_config['cell_embedding'] == 1:
        print('embedding: a true t false')
    else:  # model_config['cell_embedding'] == 2:
        print('embedding: a true t true')
    print('')

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)
    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

    model = []
    model = InteractionParticlesRollout(model_config=model_config, device=device)
    model.nstd = nstd
    model.nmean = nmean

    # state_dict = torch.load(f"./log/try_{ntry}/models/best_model.pt")
    # model.load_state_dict(state_dict['model_state_dict'])

    if model_config['cell_embedding'] == 0:
        model.a.requires_grad = False
        model.t.requires_grad = False
    if model_config['cell_embedding'] == 1:
        model.a.requires_grad = True
        model.t.requires_grad = False
    if model_config['cell_embedding'] == 2:
        model.a.requires_grad = True
        model.t.requires_grad = True

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)  # , weight_decay=5e-3)
    criteria = nn.MSELoss()
    model.train()

    print('Training model = InteractionParticles() ...')

    # ff = 0
    # trackmate = trackmate_list[ff].copy()
    # trackmate_true = trackmate_list[ff].copy()

    # ff = 0
    # trackmate = trackmate_list[ff].copy()
    trackmate_true = []

    trackmate_true = trackmate.copy()

    best_loss = np.inf

    for epoch in range(5000):

        if epoch == 100:
            optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)  # , weight_decay=5e-3)

        mserr_list = []

        trackmate = trackmate_true.copy()

        for frame in range(20, 240):  # frame_list:

            model.frame = int(frame)
            pos = np.argwhere(trackmate[:, 0] == frame)

            list_all = pos[:, 0].astype(int)
            mask = torch.tensor(np.ones(list_all.shape[0]), device=device)
            for k in range(len(mask)):
                if trackmate[list_all[k] - 1, 1] != trackmate[list_all[k] + 1, 1]:
                    mask[k] = 0
            mask = mask[:, None]

            x = torch.tensor(trackmate[list_all, 0:17], device=device)
            target = torch.tensor(trackmate_true[list_all + 1, 8:9], device=device)

            dataset = data.Data(x=x, pos=x[:, 2:4])
            transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
            dataset = transform(dataset)
            distance = dataset.edge_attr.detach().cpu().numpy()
            pos = np.argwhere(distance < model_config['radius'])
            edges = dataset.edge_index
            dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                                edge_attr=torch.tensor(distance[pos[:, 0]], device=device))

            optimizer.zero_grad()
            message, pred = model(data=dataset, data_id=0)

            loss = criteria((pred[:, :] + x[:, 8:9]) * mask, target * mask) * 3

            loss.backward()
            optimizer.step()
            mserr_list.append(loss.item())

            trackmate[list_all + 1, 10:11] = np.array(pred.detach().cpu())
            trackmate[list_all + 1, 8:9] = trackmate[list_all, 8:9] + trackmate[list_all + 1, 10:11]

        print(f"Epoch: {epoch} Loss: {np.round(np.mean(mserr_list), 4)}")

        if (np.mean(mserr_list) < best_loss):
            best_loss = np.mean(mserr_list)
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', 'best_model_new.pt'))

    print(' end of training')
    print(' ')

if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(f'Device :{device}')

    # file_list=["/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/",\
    #            "/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210108/",\
    #            "/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210109/"]
    # file_folder = file_list[0]
    # print(file_folder)
    #
    # file_list=["/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/",\
    #            "/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210108/",\
    #            "/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210109/"]
    # file_folder = file_list[0]
    # print(file_folder)
    #
    # print('Loading trackmate ...')
    # trackmate_list=[]
    # for ff in range(1):
    #     trackmate = np.load(f'{file_list[ff]}/trackmate/transformed_spots_try{415+ff}.npy')
    #     trackmate[-1, 0] = -1
    #     trackmate_list.append(trackmate)
    #     if ff==0:
    #         n_tracks = np.max(trackmate[:, 1]) + 1
    #         trackmate_true = trackmate.copy()
    #         nstd = np.load(f'{file_folder}/trackmate/nstd_try415.npy')
    #         nmean = np.load(f'{file_folder}/trackmate/nmean_try415.npy')
    #         c = nstd[6] / nstd[2]
    #
    # print('Trackmate quality check...')
    # time.sleep(0.5)
    # for ff in range(1):
    #     trackmate = trackmate_list[ff]
    #     for k in tqdm(range(5, trackmate.shape[0] - 1)):
    #         if trackmate[k-1, 1] == trackmate[k+1, 1]:
    #
    #             if np.abs(trackmate[k+1, 6] * c - (trackmate[k+1, 2] - trackmate[k, 2])) > 1E-3:
    #                 print(f'Pb check vx at row {k}')
    #             if np.abs(trackmate[k+1, 7] * c - (trackmate[k+1, 3] - trackmate[k, 3])) > 1E-3:
    #                 print(f'Pb check vy at row {k}')
    #
    #             if np.abs(trackmate[k+1, 15] - (trackmate[k+1, 6] - trackmate[k, 6])) > 1E-3:
    #                 print(f'Pb check accx at row {k}')
    #             if np.abs(trackmate[k+1, 16] - (trackmate[k+1, 7] - trackmate[k, 7])) > 1E-3:
    #                 print(f'Pb check accy at row {k}')
    # print('... done')

    # training_list=[[370,0,1,0, 1], [371,0,2,0, 1], [372,0,3,0, 1]] # rot_mode=1 cell embedding
    # training_list=[[373, 0, 1, 1, 1], [374, 0, 2, 1, 1], [375, 0, 3, 1, 1]] # rot_mode=1 no cell embedding
    # training_list=[[376, 0, 1, 2, 1], [377, 0, 2, 2, 1], [378, 0, 3, 2, 1]] # rot_mode=1 cell + time embedding
    # training_list=[[379,0,1,0,0], [380,0,2,0,0], [381,0,3,0,0]] # rot_mode=0 cell embedding
    # training_list = [[382,1,3,0,0], [383,1,1,0,0]] # no add term rot_mode=0 cell embedding
    # training_list = [[384,1,1,1,0], [385,1,1,0,1], [386,1,1,1,1]] # no add term rot_mode=0 no cell embedding
    # training_list = [[387, 0, 2, 1, 0], [388, 0, 3, 1, 0]] # rot_mode=0 no cell embedding
    # training_list = np.array(training_list)


    model_config = {'ntry': 470,
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

    model_config = {'ntry': 501,
                    'datum': '2309012_490',
                    'trackmate_metric' : {'Label': 0,
                    'Spot_ID': 1,
                    'Track_ID': 2,
                    'Quality': 3,
                    'X': 4,
                    'Y': 5,
                    'Z': 6,
                    'T': 7,
                    'Frame': 8,
                    'R': 9,
                    'Visibility': 10,
                    'Spot color': 11,
                    'Mean Ch1': 12,
                    'Median Ch1': 13,
                    'Min Ch1': 14,
                    'Max Ch1': 15,
                    'Sum Ch1': 16,
                    'Std Ch1': 17,
                    'Ctrst Ch1': 18,
                    'SNR Ch1': 19,
                    'El. x0': 20,
                    'El. y0': 21,
                    'El. long axis': 22,
                    'El. sh. axis': 23,
                    'El. angle': 24,
                    'El. a.r.': 25,
                    'Area': 26,
                    'Perim.': 27,
                    'Circ.': 28,
                    'Solidity': 29,
                    'Shape index': 30},

                    'metric_list' : ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch1', 'Area'],

                    'file_folder' : '/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/',

                    'dx':0.908,
                    'dt':5.0,
                    'h': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode':1,
                    'embedding': 8,
                    'cell_embedding': 1,
                    'time_embedding': False,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'bNoise': False,
                    'noise_level': 0,
                    'batch_size': 4,
                    'bRollout': False,
                    'rollout_window': 2,
                    'frame_start': 20,
                    'frame_end': [241],
                    'n_tracks': 0,
                    'radius': 0.15}

    trackmate_metric = model_config['trackmate_metric']
    print('')
    ntry = model_config['ntry']
    print(f'ntry: {ntry}')
    datum = model_config['datum']
    print(f'datum: {datum}')
    file_folder = model_config['file_folder']
    print(f'file_folder: {file_folder}')
    dx = model_config['dx']
    print(f'dx: {dx} microns')
    dt = model_config['dt']
    print(f'dt: {dt} minutes')
    metric_list = model_config['metric_list']
    print(f'metric_list: {metric_list}')
    frame_end = model_config['frame_end']
    print(f'frame_end: {frame_end}')

    folder = f'./graphs_data/graphs_cells_{datum}/'

    print('Loading trackmate ...')
    trackmate_list=[]

    print('Loading trackmate file ...')
    trackmate = np.load(f'{folder}/transformed_spots.npy')
    trackmate[-1, 0] = -1
    nstd = np.load(f'{folder}/nstd.npy')
    nmean = np.load(f'{folder}/nmean.npy')
    c0 = nstd[4] / nstd[2] * dt
    c1 = nstd[6] / nstd[4]
    print('done ...')
    n_tracks = np.max(trackmate[:, 1])
    model_config['n_tracks'] = n_tracks+1
    print(f'n_tracks: {n_tracks}')

    print('Trackmate quality check...')
    time.sleep(0.5)

    for k in tqdm(range(1, trackmate.shape[0] - 1)):
        if trackmate[k-1, 1] == trackmate[k+1, 1]:

            if np.abs(trackmate[k+1, 4] * c0 - (trackmate[k+1, 2] - trackmate[k, 2])) > 1E-3:
                print(f'Pb check vx at row {k}')
            if np.abs(trackmate[k+1, 5] * c0 - (trackmate[k+1, 3] - trackmate[k, 3])) > 1E-3:
                print(f'Pb check vy at row {k}')

            if np.abs(trackmate[k+1, 6] * c1 - (trackmate[k+1, 4] - trackmate[k, 4])) > 1E-3:
                print(f'Pb check accx at row {k}')
            if np.abs(trackmate[k+1, 7] * c1 - (trackmate[k+1, 5] - trackmate[k, 5])) > 1E-3:
                print(f'Pb check accy at row {k}')

    print('... done')

    print('')
    print(f'x {np.round(nmean[2], 1)}+/-{np.round(nstd[2], 1)}')
    print(f'y {np.round(nmean[3], 1)}+/-{np.round(nstd[3], 1)}')
    print(f'vx {np.round(nmean[4], 4)}+/-{np.round(nstd[4], 4)}')
    print(f'vy {np.round(nmean[5], 4)}+/-{np.round(nstd[5], 4)}')
    print(f'ax {np.round(nmean[6], 4)}+/-{np.round(nstd[6], 4)}')
    print(f'ay {np.round(nmean[7], 4)}+/-{np.round(nstd[7], 4)}')
    print(f'signal 1 {np.round(nmean[8], 2)}+/-{np.round(nstd[8], 2)}')
    print(f'signal 2 {np.round(nmean[9], 2)}+/-{np.round(nstd[9], 2)}')
    print(f'signal 1 deriv {np.round(nmean[10], 2)}+/-{np.round(nstd[10], 2)}')
    print(f'signal 2 deriv {np.round(nmean[11], 2)}+/-{np.round(nstd[11], 2)}')
    print(f'degree {np.round(nmean[16], 2)}+/-{np.round(nstd[16], 2)}')
    print('')



    train_model(model_config, trackmate)

    # model_config = {'ntry': 471,
    #                 'h': 0,
    #                 'msg': 2,
    #                 'aggr': 0,
    #                 'rot_mode':1,
    #                 'embedding': 3,
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

    # model_config = {'ntry': 472,
    #                 'h': 0,
    #                 'msg': 3,
    #                 'aggr': 0,
    #                 'rot_mode':1,
    #                 'embedding': 3,
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
    # model_config = {'ntry': 373,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 3,
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
    # model_config = {'ntry': 374,
    #                 'h': 0,
    #                 'msg': 2,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 3,
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
    # model_config = {'ntry': 375,
    #                 'h': 0,
    #                 'msg': 3,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 3,
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
    # model_config = {'ntry': 376,
    #                 'h': 0,
    #                 'msg': 4,
    #                 'aggr': 0,
    #                 'rot_mode': 0,
    #                 'embedding': 3,
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





