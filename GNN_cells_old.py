
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


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers,
                 device):  # in_feats=2, out_feats=2, num_layers=3, hidden=128):

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

class InteractionCells(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, model_config, device):  # in_feats=9, out_feats=2, num_layers=2, hidden=16):

        super(InteractionCells, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        self.device = device
        self.h = model_config['upgrade']
        self.msg = model_config['msg']
        
        self.cell_embedding = model_config['cell_embedding']
        self.dim_embedding = model_config['dim_embedding']
        self.nlayers = model_config['n_mp_layers']
        self.hidden_size = model_config['hidden_size']
        self.noise_level = model_config['noise_level']
        self.n_tracks = model_config['n_tracks']
        self.rot_mode = model_config['rot_mode']

        if (self.msg == 0):
            self.lin_edge = MLP(input_size=1 + self.dim_embedding, hidden_size=self.hidden_size, output_size=1,
                                layers=self.nlayers, device=self.device)
        if (self.msg == 1):
            self.lin_edge = MLP(input_size=5 + self.dim_embedding, hidden_size=self.hidden_size, output_size=1,
                                layers=self.nlayers, device=self.device)
        if (self.msg == 2):
            self.lin_edge = MLP(input_size=6 + self.dim_embedding, hidden_size=self.hidden_size, output_size=1,
                                layers=self.nlayers, device=self.device)

        self.lin_update = MLP(input_size=1, hidden_size=self.hidden_size, output_size=1, layers=self.nlayers, device=self.device)

        self.a = nn.Parameter(torch.tensor(np.ones((int(self.n_tracks+1), 2)), device=self.device, requires_grad=True))
        self.t = nn.Parameter(torch.tensor(np.ones((int(self.n_tracks+1), 2)), device=self.device, requires_grad=True))
        self.b = nn.Parameter(torch.tensor(np.ones((int(self.n_tracks+1), 2)), device=self.device, requires_grad=True))

    def forward(self, data):


        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        phi = torch.randn([x.shape[0]], dtype=torch.float32, requires_grad=False, device='cuda:0') * np.pi * 2
        cos = torch.cos(phi)
        sin = torch.sin(phi)

        x = torch.cat((x, cos[:, None], sin[:, None]), axis=-1)

        message = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        # derivative of ERK activity
        cell_id = x[:, 1].detach().cpu().numpy()
        rate = self.b[cell_id]

        if self.h == 0:
            pred = self.lin_update(message[:, 0:1]) - rate[:, 0:1] * (x[:, 8:9] - rate[:, 1:2])
        elif self.h == 1:
            pred = self.lin_update(message[:, 0:1])

        return message, pred

    def message(self, x_i, x_j, edge_attr):

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

        if self.msg==2:
            diff_erk = x_j[:, 4:5] / torch.clamp(torch.sqrt((x_i[:, 16:17] * nstd[16:17] + nmean[16:17]) * (x_j[:, 16:17] * nstd[16:17] + nmean[16:17])), min=1)

        d = edge_attr

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

        if self.msg == 0:
            return diff_erk * 0
        elif self.msg == 1:
            in_features = torch.cat((x_jp , y_jp , vx_p, vy_p, d), dim=-1)
        elif self.msg == 2:
            in_features = torch.cat((x_jp, y_jp, vx_p, vy_p, d, diff_erk), dim=-1)

        if self.cell_embedding:
            cell_id = x_i[:, 1].detach().cpu().numpy()
            cell_embedding = self.a[cell_id]
            x_i_type_0 = cell_embedding[:, 0]
            x_i_type_1 = cell_embedding[:, 1]
            cell_embedding = torch.cat((x_i_type_0[:, None].repeat(1, 4),x_i_type_1[:, None].repeat(1, 4)), dim=-1)
            out = self.lin_edge(torch.cat((in_features, cell_embedding), dim=-1))
        else:
            out = self.lin_edge(in_features)

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)


if __name__ == "__main__":


    model_lin = LinearRegression()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Device :{device}')

    model_config = {'ntry': 600,
                    'datum': '2309012_600',
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
                                        'Mean Ch2': 18,
                                        'Median Ch2': 19,
                                        'Min Ch2': 20,
                                        'Max Ch2': 21,
                                        'Sum Ch2': 22,
                                        'Std Ch2': 23,
                                        'Ctrst Ch1': 24,
                                        'SNR Ch1': 25,
                                        'Ctrst Ch2': 26,
                                        'SNR Ch2': 27,
                                        'El. x0': 28,
                                        'El. y0': 29,
                                        'El. long axis': 30,
                                        'El. sh. axis': 31,
                                        'El. angle': 32,
                                        'El. a.r.': 33,
                                        'Area': 34,
                                        'Perim.': 35,
                                        'Circ.': 36,
                                        'Solidity': 37,
                                        'Shape index': 38},

                    'metric_list' : ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch2', 'Mean Ch1'],

                    'file_folder' : '/home/allierc@hhmi.org/Desktop/signaling/H2B-ERK signaling/',

                    'dx':0.8,
                    'dt':6.7,
                    'n_tracks':1240,
                    'frame_start': 0,
                    'frame_end' : 88,
                    'aggr_type': 'add',
                    'upgrade': 0,
                    'msg': 1,
                    'rot_mode': 1,
                    'cell_embedding': False,
                    'dim_embedding': 0,
                    'time_embedding': False,
                    'n_mp_layers': 5,
                    'hidden_size': 256,
                    'bNoise': False,
                    'noise_level': 0,
                    'batch_size': 4,
                    'bRollout': True,
                    'rollout_window': 2,
                    'radius': 0.05,
                    }

    scaler = StandardScaler()

    for step in range(1,2):

        if step ==0:

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

            file_folder='/home/allierc@hhmi.org/Desktop/signaling/H2B-ERK signaling/'


            trackmate_csv = pd.read_csv(f'{file_folder}/spots.csv', header=3, usecols=trackmate_metric.values(), names=trackmate_metric.keys())
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

            # plt.ion()
            # frame=20
            # I = imread('/home/allierc@hhmi.org/Desktop/signaling/H2B-ERK signaling/elife-78837-fig4-video1-ERK.tif')
            # I = np.array(I)
            # t = np.squeeze(I[frame, :, :])
            # plt.imshow(t)
            # pos = np.argwhere(trackmate[:, 0] == frame)
            # xx0 = trackmate[pos, 2:4]
            # xx0 = np.squeeze(xx0)
            # plt.scatter(xx0[:, 0], xx0[:, 1], s=4, marker='.', c='r')

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
                        # plt.plot( trackmate[ppos[p1], 2],trackmate[ppos[p1], 3],'+')
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

            time.sleep(0.5)
            new_trackmate = trackmate[0:1, :]
            for p in tqdm(range(1, trackmate.shape[0] - 1)):

                if (trackmate[p, 1] == trackmate[p+1, 1]) & (trackmate[p + 1, 0] != trackmate[p, 0] + 1):
                    gap = trackmate[p + 1, 0] - trackmate[p, 0] - 1
                    first = trackmate[p:p + 1, :]
                    last = trackmate[p + 1:p + 2, :]
                    step_ = (last - first) / (gap + 1)
                    step_[0, 0] = 1
                    step_[0, flag_column] = 0
                    step_[0, 10] = 0   # do not interpolate degree
                    new_trackmate = np.concatenate((new_trackmate, trackmate[p:p + 1, :]), axis=0)
                    for k in range(gap.astype(int)):
                        new_trackmate = np.concatenate((new_trackmate, new_trackmate[-1:, :] + step_), axis=0)
                else:
                    new_trackmate = np.concatenate((new_trackmate, trackmate[p:p + 1, :]), axis=0)

            trackmate = new_trackmate

            print('Transform table ...')

            new_trackmate = np.concatenate((trackmate[:,0:4],trackmate[:,6:8],trackmate[:,15:17],trackmate[:,4:6],trackmate[:,8:10],trackmate[:,11:15],trackmate[:,10:11],trackmate[:,17:19]),axis=1)

            trackmate = np.concatenate((new_trackmate, np.zeros((trackmate.shape[0], 15))), axis=1)

            print('Derivative ...')

            time.sleep(0.5)

            trackmate[:,2:4] = trackmate[:,2:4] / dx    # conversion to microns

            flag_column = trackmate.shape[1] - 1
            n_tracks = np.max(trackmate[:, 1])+1

            # derivative calculation

            n_list=[2,3,8,9]
            for n in n_list:
                diff = trackmate[1:, n] - trackmate[:-1, n]
                diff = np.concatenate((np.zeros(1), diff))
                trackmate[:, n+2] = diff
            trackmate[:, 4:6] = trackmate[:, 4:6] / dt  # conversion to microns/minutes
            n_list=[4,5]
            for n in n_list:
                diff = trackmate[1:, n] - trackmate[:-1, n]
                diff = np.concatenate((np.zeros(1), diff))
                trackmate[:, n+2] = diff
            n_list=[14]
            for n in n_list:
                diff = trackmate[1:, n] - trackmate[:-1, n]
                diff = np.concatenate((np.zeros(1), diff))
                trackmate[:, n+1] = diff

            for k in range(5,trackmate.shape[0]):
                if trackmate[k-1,1]!=trackmate[k,1]:
                    trackmate[k, 4:8] = 0
                    trackmate[k, 10:12] = 0
                    trackmate[k, 15:16] = 0

            for k in range(1, trackmate.shape[0] - 1):
                if trackmate[k-1 , 1] == trackmate[k, 1]:
                    trackmate[k, 17] = 1  # flag pred-1
                if trackmate[k + 1 , 1] == trackmate[k, 1]:
                    trackmate[k, 18] = 1  # flag pred+1

            # normalization

            temp = trackmate[:,17]+trackmate[:,18]
            pos = np.argwhere(temp==2)
            f_trackmate=trackmate[pos,:]

            nmean = np.squeeze(np.mean(f_trackmate, axis=0))
            nstd = np.squeeze(np.std(f_trackmate, axis=0))

            nstd[3] = nstd[2]  # x and y
            nstd[5] = nstd[4]  # vx and vy
            nstd[7] = nstd[6] # accx and accy

            nstd[10] = nstd[8]  # signal 1 and its derivative
            nstd[11] = nstd[9]  # signal 2 and its derivative


            print('')
            print(f'x {np.round(nmean[2],1)}+/-{np.round(nstd[2],1)}')
            print(f'y {np.round(nmean[3], 1)}+/-{np.round(nstd[3], 1)}')
            print(f'vx {np.round(nmean[4],4)}+/-{np.round(nstd[4],4)}')
            print(f'vy {np.round(nmean[5], 4)}+/-{np.round(nstd[5], 4)}')
            print(f'ax {np.round(nmean[6],4)}+/-{np.round(nstd[6],4)}')
            print(f'ay {np.round(nmean[7], 4)}+/-{np.round(nstd[7], 4)}')
            print(f'signal 1 {np.round(nmean[8],2)}+/-{np.round(nstd[8],2)}')
            print(f'signal 2 {np.round(nmean[9], 2)}+/-{np.round(nstd[9], 2)}')
            print(f'degree {np.round(nmean[16], 2)}+/-{np.round(nstd[16], 2)}')
            print('')

            # normalization

            trackmate[:, 2:17] = (trackmate[:, 2:17] - nmean[2:17]) / nstd[2:17]

            print ('Fillling past and future ...')

            time.sleep(0.5)

            for k in tqdm(range(5,trackmate.shape[0]-1)):

                if trackmate[k + 1 , 1] == trackmate[k, 1]:
                    trackmate[k, 22:32] = trackmate[k + 1, 2:12]

            print('Trackmate quality check...')

            time.sleep(0.5)

            for k in tqdm(range(5, trackmate.shape[0] - 1)):
                if trackmate[k-1, 1] == trackmate[k+1, 1]:
                    if np.abs((trackmate[k+1, 4]*nstd[4] + nmean[4])*dt - (trackmate[k+1, 2] - trackmate[k, 2])*nstd[2]) > 1E-3:
                        print(f'Pb check vx at row {k}')
                    if np.abs((trackmate[k+1, 5]*nstd[5] + nmean[5])*dt - (trackmate[k+1, 3] - trackmate[k, 3])*nstd[3]) > 1E-3:
                        print(f'Pb check vy at row {k}')
                    if np.abs((trackmate[k+1, 6]*nstd[6] + nmean[6]) - (trackmate[k+1, 4] - trackmate[k, 4])*nstd[4]) > 1E-3:
                        print(f'Pb check accx at row {k}')
                    if np.abs((trackmate[k+1, 7]*nstd[7] + nmean[7]) - (trackmate[k+1, 5] - trackmate[k, 5])*nstd[5]) > 1E-3:
                        print(f'Pb check accy at row {k}')

            print('Check done')

            print('Saving data ...')

            trackmate = np.concatenate((trackmate, np.zeros((1, trackmate.shape[1]))), axis=0)
            trackmate[-1, 0] = -1
            np.save(f'{folder}/transformed_spots.npy', trackmate)
            np.save(f'{folder}/nstd.npy', nstd)
            np.save(f'{folder}/nmean.npy', nmean)

        if step ==1:

            files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/CellGraph/tmp_training/*")
            for f in files:
                os.remove(f)

            print('')
            print('Training loop ...')
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
            frame_start = model_config['frame_start']
            print(f'frame_start: {frame_start}')
            frame_end = model_config['frame_end']
            print(f'frame_end: {frame_end}')
            print('')
            if model_config['upgrade'] == 0:
                print('H: MLP(message - b Erk)')
            else:  # model_config['h'] == 1:
                print('H: MLP(message)')
            aggr_type = model_config['aggr_type']
            print(f'aggr_type: {aggr_type}')
            if model_config['msg'] == 0:
                print('msg: 0')
            elif model_config['msg'] == 1:
                print('msg: MLP(x_jp, y_jp, vx_p, vy_p, d, deg)')
            elif model_config['msg'] == 2:
                print('msg: MLP(x_jp, y_jp, vx_p, vy_p, d, deg, diff_erk)')
            else:  # model_config['msg']==4:
                print('msg: 0')
            print(f'rot_mode:', model_config['rot_mode'])
            radius=model_config['radius']
            print(f'radius:', model_config['radius'])
            print('')
            l_dir = os.path.join('.', 'log')
            log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
            print('log_dir: {}'.format(log_dir))

            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
            os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)

            copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

            folder = f'./graphs_data/graphs_cells_{datum}/'

            print('Loading trackmate file ...')
            trackmate = np.load(f'{folder}/transformed_spots.npy')
            nstd = np.load(f'{folder}/nstd.npy')
            nmean = np.load(f'{folder}/nmean.npy')
            print('done ...')
            model_config['n_tracks'] = np.max(trackmate[:, 1])
            print(f'n_tracks: {np.max(trackmate[:, 1])}')

            model = InteractionCells(model_config=model_config, device=device)

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
            print('')

            model.a.requires_grad = model_config['cell_embedding']
            model.t.requires_grad = model_config['time_embedding']

            optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)  # , weight_decay=5e-3)
            criteria = nn.MSELoss()
            model.train()

            print('Training model = InteractionCells() ...')

            trackmate_true = trackmate.copy()
            best_loss = np.inf

            optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)  # , weight_decay=5e-3)
            criteria = nn.MSELoss()
            model.train()

            list_loss = []
            g_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

            for epoch in range(10000):

                if epoch == 100:
                    optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)  # , weight_decay=5e-3)

                mserr_list = []

                trackmate = trackmate_true.copy()

                for frame in range(frame_start, frame_end-1):  # frame_list:

                    model.frame = int(frame)
                    pos = np.argwhere(trackmate[:, 0] == frame)

                    list_all = pos[:, 0].astype(int)
                    mask = torch.tensor(trackmate_true[list_all, 18:19], device=device)
                    mask = mask[:, None]

                    x = torch.tensor(trackmate[list_all, 0:17], device=device)
                    target = torch.tensor(trackmate_true[list_all, 30:31], device=device)       # prediction of derivative of ERK

                    dataset = data.Data(x=x, pos=x[:, 2:4])
                    transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
                    dataset = transform(dataset)
                    distance = dataset.edge_attr.detach().cpu().numpy()
                    pos = np.argwhere(distance < model_config['radius'])
                    edges = dataset.edge_index
                    dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                                        edge_attr=torch.tensor(distance[pos[:, 0]], device=device))

                    optimizer.zero_grad()
                    message, pred = model(data=dataset)

                    pos = torch.argwhere(mask==1)

                    loss = (pred[pos[:,0],:] - target[pos[:,0],:]).norm(2)*0 + g_loss(pred[pos[:,0],:],target[pos[:,0],:])*1E8

                    loss.backward()
                    optimizer.step()
                    mserr_list.append(loss.item()/len(mask))

                    if frame == 20:
                        x_p = x
                        target_p = target
                        pred_p = pred

                    trackmate[list_all + 1, 10:11] = np.array(pred.detach().cpu())
                    trackmate[list_all + 1, 8:9] = trackmate[list_all, 8:9] + trackmate[list_all + 1, 10:11]

                print(f"Epoch: {epoch} Loss: {np.round(np.mean(mserr_list), 6)}")

                if (np.mean(mserr_list) < best_loss):
                    best_loss = np.mean(mserr_list)
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(log_dir, 'models', 'best_model.pt'))

                list_loss.append(np.mean(mserr_list))

                model.a.data = torch.clamp(model.a.data, min=-4, max=4)
                embedding = model.a.detach().cpu().numpy()
                embedding = scaler.fit_transform(embedding)
                # kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
                # sse = []
                # for k in range(1, 11):
                #     kmeans_ = KMeans(n_clusters=k, **kmeans_kwargs)
                #     kmeans_.fit(embedding)
                #     sse.append(kmeans_.inertia_)
                # kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")

                if epoch%10==0:

                    fig = plt.figure(figsize=(15, 10))
                    # plt.ion()
                    ax = fig.add_subplot(2, 3, 1)
                    plt.scatter(embedding[:, 0], embedding[:, 1], s=3)
                    plt.xlim([-4.1, 4.1])
                    plt.ylim([-4.1, 4.1])
                    plt.xlabel('Embedding 0', fontsize=12)
                    plt.ylabel('Embedding 1', fontsize=12)
                    # plt.text(-3.9, 3.6, f'kl.elbow: {kl.elbow}', fontsize=10)
                    ax = fig.add_subplot(2, 3, 2)
                    # plt.plot(range(1, 11), sse)
                    plt.xticks(range(1, 11))
                    plt.xlabel("Number of Clusters", fontsize=12)
                    plt.ylabel("SSE", fontsize=12)
                    ax = fig.add_subplot(2, 3, 3)
                    plt.plot(list_loss, color='k')
                    plt.ylabel('Loss', fontsize=10)
                    ax = fig.add_subplot(2, 3, 4)
                    plt.scatter(x_p[:, 2].detach().cpu().numpy(), x_p[:, 3].detach().cpu().numpy(), s=125, marker='.',c=target_p.detach().cpu().numpy(), vmin=-0.5, vmax=0.5)
                    plt.xlim([-2, 2])
                    plt.ylim([-2, 2])
                    ax = fig.add_subplot(2, 3, 5)
                    plt.scatter(x_p[:, 2].detach().cpu().numpy(), x_p[:, 3].detach().cpu().numpy(), s=125, marker='.',c=pred_p.detach().cpu().numpy(), vmin=-0.5, vmax=0.5)
                    plt.xlim([-2, 2])
                    plt.ylim([-2, 2])
                    ax = fig.add_subplot(2, 3, 6)
                    plt.hist(target_p.detach().cpu().numpy(),100)
                    plt.hist(pred_p.detach().cpu().numpy(),100)
                    plt.savefig(f"./tmp_training/Fig_{ntry}_{epoch}.tif")
                    plt.close()


            print(' end of training')
            print(' ')















