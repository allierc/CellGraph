
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
from pysr import PySRRegressor
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
import time
import json
from geomloss import SamplesLoss
from tifffile import imread
from prettytable import PrettyTable
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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


class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size, device=device, dtype=torch.float64
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
                # self.layers.append(torch.nn.Dropout(p=0.0))
        if layernorm:
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

class MLP2(nn.Module):
    def __init__(self, in_feats=2, out_feats=2, num_layers=3, hidden=128):

        super(MLP2, self).__init__()
        self.layers = nn.ModuleList()
        layer=nn.Linear(in_feats, hidden, device=device, dtype=torch.float64)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(layer)
        if num_layers > 2:
            for i in range(1, num_layers - 1):
                layer = nn.Linear(hidden, hidden, device=device, dtype=torch.float64)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden, out_feats, device=device, dtype=torch.float64)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(layer)

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        # x = torch.clamp(x, min=-1, max=1)
        return x

class CellConcentration(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, in_feats=9, out_feats=2, num_layers=2, hidden=16):
        super(CellConcentration, self).__init__(aggr='add')  # "Add" aggregation.

    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, dataset.edge_attr
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        out = (out + x[:, 11:12]) / (x[:, 10:11] + 1)
        return 1/out

    def message(self, x_i,x_j,edge_attr):

        return x_j[:,11:12]

class GraphEntropy(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, in_feats=9, out_feats=2, num_layers=2, hidden=16):

        super(GraphEntropy, self).__init__(aggr='mean')  # "Add" aggregation.

    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, dataset.edge_attr

        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        out=torch.clamp(out,min=1E-3)
        out = torch.sum(torch.log(out))

        # out = torch.std (out) * nstd[2] * 2.35

        return -out

    def message(self, x_i, x_j,edge_attr):

        d=torch.sqrt((x_i[:,1]-x_j[:,1])**2+(x_i[:,2]-x_j[:,2])**2)
        d=d[:,None]

        return d

    def update(self, aggr_out):

        return aggr_out     #self.lin_node(aggr_out)

class InteractionNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, hidden_size, layers):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + self.new_edges
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):

        x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        x = self.lin_edge(x)
        self.new_edges = x

        return x

class LearnedSimulator(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(
        self,
        hidden_size=128,
        n_mp_layers=5, #15, # number of GNN layers
        dim=2, # dimension of the world, typical 2D or 3D
        window_size=5, # the model looks into W frames before the frame to be predicted
    ):
        super().__init__()
        self.window_size = window_size
        self.node_in = MLP(16, hidden_size, hidden_size, 3)
        self.edge_in = MLP(dim+1, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, dim, 3, layernorm=False)
        self.n_mp_layers = n_mp_layers

        self.layer = InteractionNetwork(hidden_size, 3)

        # self.layers = torch.nn.ModuleList([InteractionNetwork(
        #     hidden_size, 3
        # ) for _ in range(n_mp_layers)])

        self.a = nn.Parameter(torch.tensor(np.ones((int(n_tracks+1), 6)), requires_grad=False, device='cuda:0'))
        self.a.requires_grad = True

    def forward(self, data):
        # pre-processing
        # node feature: combine categorial feature data.x and contiguous feature data.pos.

        track_id =data.x[:, 1].detach().cpu().numpy()
        node_feature = torch.cat((data.x[:,2:5],data.x[:,6:9],data.x[:,10:14],self.a[track_id]), dim=-1)

        # node_feature = data.x
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)
        # stack of GNN layers

        nlayers=self.n_mp_layers


        if bMotility:

            intermediate_speed=torch.tensor(np.zeros((node_feature.shape[0], 2*nlayers)), requires_grad=False, device='cuda:0')
            intermediate_speed_target=torch.tensor(np.zeros((node_feature.shape[0], 2*nlayers)), requires_grad=False, device='cuda:0')

            step_vx=(data.x[:,20]-data.x[:,6]) / nlayers
            step_vy=(data.x[:,21]-data.x[:,7]) / nlayers

            intermediate_pos = torch.tensor(np.zeros((node_feature.shape[0], 2 * nlayers)), requires_grad=False,device='cuda:0')
            intermediate_pos_target = torch.tensor(np.zeros((node_feature.shape[0], 2 * nlayers)), requires_grad=False,device='cuda:0')

            for i in range(nlayers):

                intermediate_speed_target[:, i] = data.x[:, 6] + step_vx * (i + 1) * data.x[:, 17]
                intermediate_speed_target[:, i + nlayers] = data.x[:, 7] + step_vy * (i + 1) * data.x[:, 17]
                intermediate_pos_target[:, i] = data.x[:,2] + intermediate_speed_target[:, i]*nstd[6]/nstd[2]
                intermediate_pos_target[:, i+nlayers] =data.x[:,3] + intermediate_speed_target[:, i + nlayers]*nstd[6]/nstd[2]

                node_feature, edge_feature = self.layer(node_feature, data.edge_index, edge_feature=edge_feature)
                intermediate_speed[:, i] = self.node_out(node_feature)[:, 0]
                intermediate_speed[:, i + nlayers] = self.node_out(node_feature)[:, 1]
                intermediate_pos[:,i]=data.x[:,2] + intermediate_speed[:, i]*nstd[6]/nstd[2]
                intermediate_pos[:,i+nlayers]=data.x[:,3] + intermediate_speed[:, i + nlayers]*nstd[6]/nstd[2]

            return intermediate_pos, intermediate_pos_target, intermediate_speed, intermediate_speed_target

        else:

            intermediate_pos=torch.tensor(np.zeros((node_feature.shape[0], 2*nlayers)), requires_grad=False, device='cuda:0')
            intermediate_pos_target=torch.tensor(np.zeros((node_feature.shape[0], 2*nlayers)), requires_grad=False, device='cuda:0')

            step_x=(data.x[:,18]-data.x[:,2]) / nlayers
            step_y=(data.x[:,19]-data.x[:,3]) / nlayers

            for i in range(nlayers):

                intermediate_pos_target[:, i] = data.x[:,2] + step_x*(i+1)*data.x[:,17]
                intermediate_pos_target[:, i+nlayers] = data.x[:,3] + step_y*(i+1)*data.x[:,17]

                node_feature, edge_feature = self.layer(node_feature, data.edge_index, edge_feature=edge_feature)
                intermediate_pos[:,i]=self.node_out(node_feature)[:,0]
                intermediate_pos[:,i+nlayers] = self.node_out(node_feature)[:, 1]

            return intermediate_pos, intermediate_pos_target

class InteractionParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, in_feats=9, out_feats=2, num_layers=2, hidden=16):

        super(InteractionParticles, self).__init__(aggr='add')  # "Add" aggregation.

        self.lin_edge = MLP2(in_feats=11, out_feats=1, num_layers=3, hidden=32)
        self.lin_update = MLP2(in_feats=3, out_feats=1, num_layers=2, hidden=16)

        self.a = nn.Parameter(torch.tensor(np.ones((int(n_tracks+1),5)), requires_grad=True, device='cuda:0'))
        self.b = nn.Parameter(torch.tensor(np.ones((1,1)), requires_grad=True, device='cuda:0'))

    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, dataset.edge_attr

        message = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        track_id = self.a[x[:, 1].detach().cpu().numpy()]

        out = track_id[:,0:1] * message[:,0:1] - self.b[0]*x[:,4:5] # + self.b[1]*(x[:,11:12]-x[:,10:11])

        return message, out

    def message(self, x_i, x_j,edge_attr):

        delta_pos=(x_i[:,2:4]-x_j[:,2:4])
        track_id = self.a[x_i[:, 1].detach().cpu().numpy()]

        in_features = torch.cat((delta_pos, edge_attr[:,0:1], x_i[:, 6:8], x_i[:, 11:12], x_i[:, 13:14],x_j[:, 6:8] * track_id[:, 1:3], x_j[:, 11:12]* track_id[:, 3:4], x_j[:, 13:14] * track_id[:, 4:5]), dim=-1)

        return self.lin_edge(in_features)

    def update(self, aggr_out):

        return aggr_out     #self.lin_node(aggr_out)

if __name__ == "__main__":


    model_lin = LinearRegression()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(f'Device :{device}')

    trackmate_metric = {'Label': 0,
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
                        'Shape index': 30}


    model_config = {'ntry': 470,
                    'datum': '2309012_480',
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

    print('')
    print('Generating data ...')

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
    os.makedirs(folder, exist_ok=True)

    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    json_ = json.dumps(model_config)
    f = open(f"{folder}/model_config.json", "w")
    f.write(json_)
    f.close()

    file_folder = model_config['file_folder']
    print(file_folder)

    radius = 0.05

    bMotility = True
    frame_end=240

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    print('log_dir: {}'.format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)

    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

    print (f'{file_folder}/trackmate/transformed_spots_try{ntry}.npy')

    trackmate_csv = pd.read_csv(f'{file_folder}/spots.csv', header=3, usecols=trackmate_metric.values(),names=trackmate_metric.keys())
    n = 0
    for metric in metric_list:
        track = np.array(trackmate_csv[metric])
        if n == 0:
            trackmate = track[:, None]
        else:
            trackmate = np.concatenate((trackmate, track[:, None]), axis=1)
        n += 1

    trackmate = trackmate[trackmate[:, 0].argsort()]
    trackmate = trackmate[trackmate[:, 1].argsort(kind='mergesort')]
    pos = np.isnan(trackmate[:, 1])
    pos = np.argwhere(pos == True)
    trackmate = trackmate[0:int(pos[0]), :]
    trackmate = np.concatenate((trackmate, np.zeros((trackmate.shape[0], 12))), axis=1)
    n_tracks = np.max(trackmate[:, 1])

    # frame=20
    # I = imread(f'/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210109//ACTIVITY.tif')
    # I = np.array(I)
    # t = np.squeeze(I[frame, :, :])
    # plt.imshow(t, vmin=0, vmax=2)
    # pos = np.argwhere(trackmate[:, 0] == frame)
    # xx0 = trackmate[pos, 2:4]
    # xx0 = np.squeeze(xx0)
    # xx0 = xx0 * nstd[2]
    # xx0[:, 0] = xx0[:, 0] + nmean[2]
    # xx0[:, 1] = xx0[:, 1] + nmean[3]
    # plt.scatter(xx0[:, 0], xx0[:, 1], s=125, marker='.', vmin=-0.6, vmax=0.6)

    pos = trackmate[1:, 1] - trackmate[:-1, 1]
    pos_n = np.concatenate((pos, np.ones(1)))
    pos_p = np.concatenate((np.ones(1), pos))
    flag_list = 1 - (pos_n + pos_p)
    flag_column = trackmate.shape[1] - 1
    trackmate[:, flag_column] = flag_list

    print('Voronoi calculation ...')

    model_concentration = CellConcentration()
    time.sleep(0.5)
    for frame in tqdm(range(0, frame_end)):  # frame_list:

        ppos = np.argwhere(trackmate[:, 0] == frame)

        if len(ppos) > 0:

            ppos = ppos.astype(int)
            ppos = np.squeeze(ppos)
            points = trackmate[ppos, 2:4]

            I = imread(f'{file_folder}/masks/{frame}_cp_masks.tif')
            I = np.array(I)
            I_new = I * 0

            I = gaussian_filter(I, sigma=8)

            for lx in range(I.shape[0]):
                line = I[lx, :]
                pos = np.argwhere(line > 0)
                if len(pos) > 1:
                    pos = pos.astype(int)
                    pos = np.squeeze(pos)
                    I_new[lx, pos[0]:pos[-1]] = 1

            points_out = np.zeros((1, 2))

            for lx in range(0, I.shape[0], 20):
                line = I_new[lx, :]
                pos = np.argwhere(line > 0)
                if len(pos) > 1:
                    new_point = np.array([int(pos[0]) - 10, lx])[None, :]
                    points_out = np.concatenate((points_out, new_point))
                    new_point = np.array([int(pos[-1]) + 10, lx])[None, :]
                    points_out = np.concatenate((points_out, new_point))

            for ly in range(0, I.shape[1], 20):
                line = I_new[:, ly]
                pos = np.argwhere(line > 0)
                if len(pos) > 1:
                    new_point = np.array([ly, int(pos[0]) - 10])[None, :]
                    points_out = np.concatenate((points_out, new_point))
                    new_point = np.array([ly, int(pos[-1]) + 10])[None, :]
                    points_out = np.concatenate((points_out, new_point))

            # fig = plt.figure(figsize=(25, 15))
            # plt.ion()
            # ax = fig.add_subplot(2, 2, 3)
            # plt.imshow(I_new)
            # ax = fig.add_subplot(2, 2, 1)
            # plt.imshow(I)
            # ax = fig.add_subplot(1, 2, 2)

            all_points = np.concatenate((points, points_out))
            vor = Voronoi(all_points)
            regions, vertices = voronoi_finite_polygons_2d(vor)

            def segments(poly):
                """A sequence of (x,y) numeric coordinates pairs """
                return zip(poly, poly[1:] + [poly[0]])
            def perimeter(poly):
                """A sequence of (x,y) numeric coordinates pairs """
                return abs(sum(math.hypot(x0 - x1, y0 - y1) for ((x0, y0), (x1, y1)) in segments(poly)))


            for p1, region in enumerate(regions[0:points.shape[0]]):
                polygon = vertices[region]
                trackmate[ppos[p1], 10] = len(region)      # degree
                xy_e = explode_xy(
                    polygon)  # https://www.geodose.com/2021/09/how-calculate-polygon-area-unordered-coordinates-points-python.html
                trackmate[ppos[p1], 11] = shoelace_area(xy_e[0], xy_e[1])        # polygon
                # plt.fill(*zip(*polygon), alpha=0.4)
                trackmate[ppos[p1], 12] = perimeter(polygon)

            x = torch.tensor(trackmate[ppos, 0:-1], device=device)
            dataset = data.Data(x=x, pos=x[:, 2:4])
            transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
            dataset = transform(dataset)
            distance = dataset.edge_attr.detach().cpu().numpy()
            pos = np.argwhere(distance < 100)
            edges = dataset.edge_index
            dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                                edge_attr=torch.tensor(distance[pos[:, 0]], device=device))
            pred = model_concentration(dataset)
            pred = pred.detach().cpu().numpy()
            trackmate[ppos, 13:14] = pred           # cell density

            # plt.scatter(points[:, 0], points[:, 1],c=trackmate[ppos,13])
            # plt.plot(points_out[:, 0], points_out[:, 1], 'ro')
            # plt.show()

    pos = np.isnan(trackmate[:, 13])
    pos = np.argwhere(pos == True)
    if len(pos) > 0:
        trackmate[pos, 13] = 0
    pos = np.isinf(trackmate[:, 13])
    pos = np.argwhere(pos == True)
    if len(pos) > 0:
        trackmate[pos, 13] = 0

    print ('Filling gap Trackmate ...')

    if True:
        time.sleep(0.5)
        new_trackmate = trackmate[0:1, :]
        for p in tqdm(range(1, trackmate.shape[0] - 1)):

            if (trackmate[p, 1] == trackmate[p+1, 1]) & (trackmate[p + 1, 0] != trackmate[p, 0] + 1):
                gap = trackmate[p + 1, 0] - trackmate[p, 0] - 1
                first = trackmate[p:p + 1, :]
                last = trackmate[p + 1:p + 2, :]
                step = (last - first) / (gap + 1)
                step[0, 0] = 1
                step[0, flag_column] = 0
                new_trackmate = np.concatenate((new_trackmate, trackmate[p:p + 1, :]), axis=0)
                for k in range(gap.astype(int)):
                    new_trackmate = np.concatenate((new_trackmate, new_trackmate[-1:, :] + step), axis=0)
            else:
                new_trackmate = np.concatenate((new_trackmate, trackmate[p:p + 1, :]), axis=0)
        trackmate = new_trackmate


    trackmate = np.concatenate((trackmate, trackmate[-1:, :]), axis=0)
    trackmate [-1,0]=-1

    flag_column = trackmate.shape[1] - 1
    n_tracks = np.max(trackmate[:, 1])+1

    print('Derivative ...')

    trackmate[:,2:4] = trackmate[:,2:4] * dx

    n_list=[2,3]
    for n in n_list:
        diff = trackmate[1:, n] - trackmate[:-1, n]
        diff = np.concatenate((np.zeros(1), diff))
        trackmate[:, n+4] = diff
    n_list =[4,5]
    trackmate[:, 6:8] = trackmate[:, 6:8] / dt

    for n in n_list:
        diff = trackmate[1:, n] - trackmate[:-1, n]
        diff = np.concatenate((np.zeros(1), diff))
        trackmate[:, n + 4] = diff
    n_list =[6, 7]
    for n in n_list:
        diff = trackmate[1:, n] - trackmate[:-1, n]
        diff = np.concatenate((np.zeros(1), diff))
        trackmate[:, n + 9] = diff
    n_list=[13]
    for n in n_list:
        diff = trackmate[1:, n] - trackmate[:-1, n]
        diff = np.concatenate((np.zeros(1), diff))
        trackmate[:, n+1] = diff

    for k in range(5,trackmate.shape[0]):
        if trackmate[k-1,1]!=trackmate[k,1]:
            trackmate[k, 6:10] = 0
            trackmate[k, 14] = 0
            trackmate[k, 15] = 0
            trackmate[k, 16] = 0

    trackmate = np.concatenate((trackmate, np.zeros((trackmate.shape[0], 34))), axis=1)
    trackmate[:, 17:19] = 0

    for k in range(5, trackmate.shape[0] - 1):
        if trackmate[k-5,1]==trackmate[k,1]:
            trackmate [k,17]=1 #flag pred-5
        if trackmate[k + 1 , 1] == trackmate[k, 1]:
            trackmate[k, 18] = 1  # flag pred+1

    # normalization
    temp = trackmate[:,17]+trackmate[:,18]
    pos = np.argwhere(temp==2)
    f_trackmate=trackmate[pos,:]

    nmean = np.squeeze(np.mean(f_trackmate, axis=0))
    nstd = np.squeeze(3*np.std(f_trackmate, axis=0))

    nstd[3] = nstd[2]  # x and y
    nstd[7] = nstd[6]  # vy as vx
    nstd[15] = nstd[6] # accx and vx
    nstd[16] = nstd[6] # accy and vx
    nstd[8:10] = nstd[4:6]  # signals and its derivative

    nmean[6:17] = 0

    print('')
    print(f'x {np.round(nmean[2], 1)}+/-{np.round(nstd[2], 1)}')
    print(f'y {np.round(nmean[3], 1)}+/-{np.round(nstd[3], 1)}')
    print(f'vx {np.round(nmean[4], 4)}+/-{np.round(nstd[4], 4)}')
    print(f'vy {np.round(nmean[5], 4)}+/-{np.round(nstd[5], 4)}')
    print(f'ax {np.round(nmean[6], 4)}+/-{np.round(nstd[6], 4)}')
    print(f'ay {np.round(nmean[7], 4)}+/-{np.round(nstd[7], 4)}')
    print(f'signal 1 {np.round(nmean[8], 2)}+/-{np.round(nstd[8], 2)}')
    print(f'signal 2 {np.round(nmean[9], 2)}+/-{np.round(nstd[9], 2)}')
    print(f'degree {np.round(nmean[16], 2)}+/-{np.round(nstd[16], 2)}')
    print('')

    trackmate[:, 2:17] = (trackmate[:, 2:17] - nmean[2:17]) / nstd[2:17]

    c=nstd[6]/nstd[2]*dt

    print ('Fillling past and future ...')

    time.sleep(0.5)

    for k in tqdm(range(5,trackmate.shape[0]-1)):

        if trackmate[k-5,1]==trackmate[k,1]:
            trackmate [k,17]=1 #flag pred-5

            if np.sum(trackmate[k-1,19:24])!=0:

                trackmate[k, 19:42]=trackmate[k-1, 20:43]
                trackmate[k, 23] = trackmate[k - 1, 2] #x
                trackmate[k, 28] = trackmate[k - 1, 3] #y
                trackmate[k, 33] = trackmate[k - 1, 6] #vx
                trackmate[k, 38] = trackmate[k - 1, 7] #vy
                trackmate[k, 43] = trackmate[k - 1, 4] #erk

            else:

                trackmate[k, 19:24] = trackmate[k-5:k,2]
                trackmate[k, 24:29] = trackmate[k - 5:k, 3] #y
                trackmate[k, 29:34] = trackmate[k - 5:k, 6] #vx
                trackmate[k, 34:39] = trackmate[k - 5:k, 7] #vy
                trackmate[k, 39:44] = trackmate[k - 5:k, 4] #vy


        if trackmate[k + 1 , 1] == trackmate[k, 1]:

            trackmate[k, 18] = 1  # flag pred+1

            trackmate[k, 44] = trackmate[k+1,2] #x
            trackmate[k, 45] = trackmate[k + 1, 3] #y
            trackmate[k, 46] = trackmate[k + 1, 6] #vx
            trackmate[k, 47] = trackmate[k + 1, 7] #vy
            trackmate[k, 48] = trackmate[k + 1, 15] #ax
            trackmate[k, 49] = trackmate[k + 1, 16] #ay
            trackmate[k, 50] = trackmate[k + 1, 4] #erk
            trackmate[k, 51] = trackmate[k + 1, 8] #erk deriv

        else:

            trackmate[k, 18] = 0  # flag pred+1

            trackmate[k, 44] = trackmate[k,2] #x
            trackmate[k, 45] = trackmate[k, 3] #y
            trackmate[k, 46] = 0
            trackmate[k, 47] = 0
            trackmate[k, 48] = 0
            trackmate[k, 49] = 0
            trackmate[k, 50] = trackmate[k, 4] #erk
            trackmate[k, 51] = trackmate[k, 8] #erk deriv

    print('Trackmate quality check...')

    time.sleep(0.5)

    c = nstd[6] / nstd[2] * dt

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

    for k in tqdm(range(5, trackmate.shape[0] - 1)):
        if trackmate[k-6, 1] == trackmate[k+1, 1]:

            if np.abs(trackmate[k, 6] * c - (trackmate[k, 2] - trackmate[k, 23])) > 1E-3:
                print(f'Pb check vx at row {k}')
            if np.abs(trackmate[k, 7] * c - (trackmate[k, 3] - trackmate[k, 28])) > 1E-3:
                print(f'Pb check vy at row {k}')
            if np.abs(trackmate[k, 15] - (trackmate[k, 6] - trackmate[k, 33])) > 1E-3:
                print(f'Pb check ax at row {k}')
            if np.abs(trackmate[k, 16] - (trackmate[k, 7] - trackmate[k, 38])) > 1E-3:
                print(f'Pb check ay at row {k}')
            if np.abs(trackmate[k, 8] - (trackmate[k, 4] - trackmate[k, 43])) > 1E-3:
                print(f'Pb check erk deriv at row {k}')

            if np.abs(trackmate[k, 44]-trackmate[k+1, 2])> 1E-3:
                print(f'Pb check x+1 at row {k}')
            if np.abs(trackmate[k, 45]-trackmate[k+1, 3])> 1E-3:
                print(f'Pb check y+1 at row {k}')
            if np.abs(trackmate[k, 46] - trackmate[k + 1, 6]) > 1E-3:
                print(f'Pb check vx+1 at row {k}')
            if np.abs(trackmate[k, 47] - trackmate[k + 1, 7]) > 1E-3:
                print(f'Pb check vy+1 at row {k}')
            if np.abs(trackmate[k, 48] - trackmate[k + 1, 15]) > 1E-3:
                print(f'Pb check ax+1 at row {k}')
            if np.abs(trackmate[k, 49] - trackmate[k + 1, 16]) > 1E-3:
                print(f'Pb check ay+1 at row {k}')
            if np.abs(trackmate[k, 50] - trackmate[k + 1, 4]) > 1E-3:
                print(f'Pb check ax+1 at row {k}')
            if np.abs(trackmate[k, 51] - trackmate[k + 1, 8]) > 1E-3:
                print(f'Pb check ay+1 at row {k}')

    print('Check done')

    print('Saving data ...')

    np.save(f'{folder}/transformed_spots.npy', trackmate)
    np.save(f'{folder}/nstd.npy', nstd)
    np.save(f'{folder}/nmean.npy', nmean)
