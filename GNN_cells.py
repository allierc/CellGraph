
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from shapely.geometry import MultiPoint, Point, Polygon
import seaborn as sns
import pandas as pd
import umap
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from geomloss import SamplesLoss

def load_trackmate(model_config):


    folder = model_config['dataset'][0]

    print(f'Loading trackmate file {folder} ...')
    dt = model_config['dt']
    trackmate = np.load(f'./graphs_data/graphs_cells_{folder}/transformed_spots.npy')
    trackmate[-1, 0] = -1
    nstd = np.load(f'./graphs_data/graphs_cells_{folder}/nstd.npy')
    nmean = np.load(f'./graphs_data/graphs_cells_{folder}/nmean.npy')

    c0 = nstd[4] / nstd[2] * dt
    c1 = nstd[6] / nstd[4]
    print('done ...')
    n_tracks = np.max(trackmate[:, 1])
    model_config['n_tracks'] = n_tracks + 1
    print(f'n_tracks: {n_tracks}')

    # degree no normalization
    trackmate[:,16] = trackmate[:,16] * nstd[16] + nmean[16]

    print('Trackmate quality check...')
    time.sleep(0.5)

    for k in tqdm(range(1, trackmate.shape[0] - 1)):
        if trackmate[k - 1, 1] == trackmate[k + 1, 1]:

            if np.abs(trackmate[k + 1, 4] * c0 - (trackmate[k + 1, 2] - trackmate[k, 2])) > 1E-3:
                print(f'Pb check vx at row {k}')
            if np.abs(trackmate[k + 1, 5] * c0 - (trackmate[k + 1, 3] - trackmate[k, 3])) > 1E-3:
                print(f'Pb check vy at row {k}')
            if np.abs(trackmate[k + 1, 6] * c1 - (trackmate[k + 1, 4] - trackmate[k, 4])) > 1E-3:
                print(f'Pb check accx at row {k}')
            if np.abs(trackmate[k + 1, 7] * c1 - (trackmate[k + 1, 5] - trackmate[k, 5])) > 1E-3:
                print(f'Pb check accy at row {k}')
            if np.abs(trackmate[k + 1, 10] - (trackmate[k + 1, 8] - trackmate[k, 8])) >1E-3:
                print(f'Pb check ERK at row {k}')

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

    trackmate_list = []
    trackmate_list.append(trackmate)
    n_tracks_list = []
    n_tracks_list.append(n_tracks)

    if len(model_config['dataset']) > 0:

        print ('Multiple data training ... ')

        for n in range (1,len(model_config['dataset'])):

            folder = model_config['dataset'][n]
            print(f'Loading trackmate file {folder} ...')
            trackmate = np.load(f'./graphs_data/graphs_cells_{folder}/transformed_spots.npy')
            trackmate[-1, 0] = -1
            trackmate_list.append(trackmate)
            n_tracks_list.append(np.max(trackmate[:, 1]))
            # degree no normalization
            trackmate[:, 16] = trackmate[:, 16] * nstd[16] + nmean[16]

            print('Trackmate quality check...')

            time.sleep(0.5)

            for k in tqdm(range(1, trackmate.shape[0] - 1)):
                if trackmate[k - 1, 1] == trackmate[k + 1, 1]:

                    if np.abs(trackmate[k + 1, 4] * c0 - (trackmate[k + 1, 2] - trackmate[k, 2])) > 1E-3:
                        print(f'Pb check vx at row {k}')
                    if np.abs(trackmate[k + 1, 5] * c0 - (trackmate[k + 1, 3] - trackmate[k, 3])) > 1E-3:
                        print(f'Pb check vy at row {k}')

                    if np.abs(trackmate[k + 1, 6] * c1 - (trackmate[k + 1, 4] - trackmate[k, 4])) > 1E-3:
                        print(f'Pb check accx at row {k}')
                    if np.abs(trackmate[k + 1, 7] * c1 - (trackmate[k + 1, 5] - trackmate[k, 5])) > 1E-3:
                        print(f'Pb check accy at row {k}')

            time.sleep(0.5)


    return trackmate_list, nstd, nmean, n_tracks_list

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
class InteractionParticles(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, model_config, device):   # in_feats=9, out_feats=2, num_layers=2, hidden=16):

        super(InteractionParticles, self).__init__(aggr='add')  # "Add" aggregation.

        self.device=device
        self.upgrade_type = model_config['upgrade_type']
        self.msg=model_config['msg']
        self.aggr=model_config['aggr']
        self.cell_embedding = model_config['cell_embedding']
        self.nlayers0 = model_config['n_mp_layers0']

        self.noise_level = model_config['noise_level']
        self.n_tracks = model_config['n_tracks']
        self.rot_mode = model_config['rot_mode']
        self.frame_end = model_config['frame_end']

        self.nlayers0 = model_config['n_mp_layers0']
        self.hidden_size0 = model_config['hidden_size0']
        self.output_size0 = model_config['output_size0']

        self.nlayers1 = model_config['n_mp_layers1']
        self.hidden_size1 = model_config['hidden_size1']
        self.time_window = model_config['time_window']

        if (self.msg == 0):
            self.lin_edge = MLP(input_size=1, hidden_size=self.hidden_size0, output_size=self.output_size0, layers=self.nlayers0, device=self.device)
        if (self.msg==1) | (self.msg==2):
            self.lin_edge = MLP(input_size=8, hidden_size=self.hidden_size0, output_size=self.output_size0, layers=self.nlayers0, device=self.device)

        self.lin_update = MLP(input_size=self.output_size0+self.cell_embedding + 2 + self.time_window, hidden_size=self.hidden_size1, output_size=1, layers=self.nlayers1, device=self.device)


        self.a = nn.Parameter(torch.tensor(np.ones((len(model_config['dataset']),int(self.n_tracks+1), self.cell_embedding)), requires_grad=True, device=self.device))
        # self.t = nn.Parameter(torch.tensor(np.ones((len(model_config['dataset']),int(self.n_tracks+1), 1)), requires_grad=False, device=self.device))


    def forward(self, data, data_id, step, cos_phi, sin_phi):

        self.data_id = data_id
        self.step = step
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        message = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

        # derivative of ERK activity
        cell_id = x[:, 1].detach().cpu().numpy()
        embedding = self.a[self.data_id,cell_id]

        pred = self.lin_update(torch.cat((embedding, x[:, 8:9], message, x[:, 17:17 + self.time_window]), axis=-1))

        return pred

    def message(self, x_i, x_j,edge_attr):

        delta_pos = x_j[:, 2:4] - x_i[:, 2:4]
        x_i_vx = x_i[:, 4:5] 
        x_i_vy = x_i[:, 5:6] 
        x_j_vx = x_j[:, 4:5] 
        x_j_vy = x_j[:, 5:6] 

        diff_erk = x_j[:, 8:9] - x_i[:, 8:9]
        erk_j = x_j[:, 8:9]

        if (self.rot_mode == 1) & (self.step == 1):

            new_x = self.cos_phi * delta_pos[:,0] + self.sin_phi * delta_pos[:,1]
            new_y = -self.sin_phi * delta_pos[:,0] + self.cos_phi * delta_pos[:,1]
            delta_pos[:,0] = new_x
            delta_pos[:,1] = new_y
            new_vx = self.cos_phi * x_i_vx + self.sin_phi * x_i_vy
            new_vy = -self.sin_phi * x_i_vx + self.cos_phi * x_i_vy
            x_i_vx = new_vx
            x_i_vy = new_vy
            new_vx = self.cos_phi * x_j_vx + self.sin_phi * x_j_vy
            new_vy = -self.sin_phi * x_j_vx + self.cos_phi * x_j_vy
            x_j_vx = new_vx
            x_j_vy = new_vy

        d = edge_attr

        if self.msg==0:
            return diff_erk*0
        elif self.msg==1:
            in_features = torch.cat((delta_pos, d, x_i_vx, x_i_vy, x_j_vx, x_j_vy, diff_erk), dim=-1)
        elif self.msg==2:
            in_features = torch.cat((delta_pos, d, x_i_vx, x_i_vy, x_j_vx, x_j_vy, erk_j), dim=-1)

        out = self.lin_edge(in_features)

        return out


    def update(self, aggr_out):

        return aggr_out     #self.lin_node(aggr_out)

def train_model_Interaction(model_config=None, trackmate_list=None, nstd=None, nmean=None, n_tracks_list=None):

    ntry = model_config['ntry']
    data_augmentation = model_config['rot_mode']>0
    batch_size = model_config['batch_size']
    training_mode = model_config['training_mode']
    nDataset = len(model_config['dataset'])
    time_window = model_config['time_window']
    batch_size = 1
    print(f'batchsize: {batch_size}')
    print('')

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)
    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

    model = []
    model = InteractionParticles(model_config=model_config, device=device)
    model.nstd = nstd
    model.nmean = nmean

    # state_dict = torch.load(f"./log/try_{ntry}/models/best_model_new.pt")
    # model.load_state_dict(state_dict['model_state_dict'])

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

    lr = 1E-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # , weight_decay=5e-3)
    print(f'Learning rates: {lr}')
    criteria = nn.MSELoss()
    model.train()

    print('Training model = InteractionParticles() ...')
    print(' ')

    data_id = 0
    trackmate = trackmate_list[data_id].copy()
    trackmate_true = trackmate_list[data_id].copy()

    best_loss = np.inf

    if data_augmentation:
        data_augmentation_loop = 20
        print(f'data_augmentation_loop: {data_augmentation_loop}')
    else:
        data_augmentation_loop = 1
        print('no data augmentation ...')

    list_plot = []


    for epoch in range(60):

        if epoch == 5:
            batch_size = model_config['batch_size']
            print(f'batchsize: {batch_size}')
        if epoch % 5 == 0:
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       f'{log_dir}/models/model_new_{epoch}.pt')
        if epoch == 20:
            lr=1E-4
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # , weight_decay=5e-3)
            print(f'Learning rates: {lr}')
            data_augmentation_loop = 200
            print(f'data_augmentation_loop: {data_augmentation_loop}')

        loss_list = []

        if training_mode == 't+1':
            for N in range(1, np.sum(model_config['frame_end']) * data_augmentation_loop // batch_size):

                data_id = np.random.randint(nDataset)
                dataset_batch = []
                mask_batch = []

                for batch in range(batch_size):

                    phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                    cos_phi = torch.cos(phi)
                    sin_phi = torch.sin(phi)

                    m = 2 + time_window + np.random.randint(model_config['frame_end'][data_id] - 2)

                    pos = np.argwhere(trackmate_list[data_id][:, 0] == m)
                    list_all = pos[:, 0].astype(int)
                    mask = torch.tensor(np.ones(list_all.shape[0]), device=device)
                    for k in range(len(mask)):
                        if trackmate[list_all[k] - 1 - time_window, 1] != trackmate[list_all[k] + 1, 1]:
                            mask[k] = 0
                    mask = mask[:, None]
                    if batch==0:
                        mask_batch=mask
                    else:
                        mask_batch=torch.cat((mask_batch, mask), axis=0)

                    x = torch.tensor(trackmate_list[data_id][list_all, 0:17], device=device)
                    if time_window>0:
                        for k in range(time_window):
                            x=torch.cat((x,torch.tensor(trackmate_list[data_id][list_all-k-1, 8:9], device=device)),axis=-1)
                    if model_config['noise_level'] > 0:
                        noise_current = torch.randn((x.shape[0], 2), device=device) * model_config['noise_level']
                        x[:,2:4] = x[:,2:4] + noise_current

                    dataset = data.Data(x=x, pos=x[:, 2:4])
                    transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
                    dataset = transform(dataset)
                    distance = dataset.edge_attr.detach().cpu().numpy()
                    pos = np.argwhere(distance < model_config['radius'])
                    edges = dataset.edge_index
                    dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                                        edge_attr=torch.tensor(distance[pos[:, 0]], device=device))
                    dataset_batch.append(dataset)

                    y = torch.tensor(trackmate_list[data_id][list_all+1, 10:11], device=device)
                    if batch==0:
                        y_batch=y
                    else:
                        y_batch=torch.cat((y_batch, y), axis=0)

                batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                optimizer.zero_grad()

                for batch in batch_loader:
                    pred = model(batch, data_id=data_id, step=1, cos_phi=cos_phi, sin_phi=sin_phi)

                loss = ((pred - y_batch)*mask_batch).norm(2)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item()/batch_size)

        elif training_mode == 'regressive':

            regressive_step = 2

            data_id = np.random.randint(nDataset)
            trackmate = trackmate_list[data_id].copy()
            trackmate_true = trackmate_list[data_id].copy()

            for N in range(1,model_config['frame_end'][data_id] * data_augmentation_loop //regressive_step):  # frame_list:

                m = 2 + time_window + np.random.randint(model_config['frame_end'][data_id] - - 1 - regressive_step)

                optimizer.zero_grad()
                loss=0

                phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
                cos_phi = torch.cos(phi)
                sin_phi = torch.sin(phi)

                for regressive_loop in range(regressive_step):

                    pos = np.argwhere(trackmate[:, 0] == m + regressive_loop)
                    list_all = pos[:, 0].astype(int)
                    mask = torch.tensor(np.ones(list_all.shape[0]), device=device)
                    for k in range(len(mask)):
                        if trackmate[list_all[k] - 1 - time_window, 1] != trackmate[list_all[k] + 1, 1]:
                            mask[k] = 0
                    mask = mask[:, None]

                    x = torch.tensor(trackmate[list_all, 0:17], device=device)
                    if time_window>0:
                        for k in range(time_window):
                            x=torch.cat((x,torch.tensor(trackmate_list[data_id][list_all-k-1, 8:9], device=device)),axis=-1)
                    target = torch.tensor(trackmate_true[list_all + 1, 10:11], device=device)

                    dataset = data.Data(x=x, pos=x[:, 2:4])
                    transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
                    dataset = transform(dataset)
                    distance = dataset.edge_attr.detach().cpu().numpy()
                    pos = np.argwhere(distance < model_config['radius'])
                    edges = dataset.edge_index
                    dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                                        edge_attr=torch.tensor(distance[pos[:, 0]], device=device))

                    pred = model(data=dataset, data_id=data_id, step=1, cos_phi=cos_phi, sin_phi=sin_phi)

                    loss += ((pred-target)*mask).norm(2)

                    for k in range(len(mask)):
                        if mask[k] == 1:
                            trackmate[list_all[k] + 1, 10:11] = np.array(pred[k].detach().cpu())
                            trackmate[list_all[k] + 1, 8:9] = trackmate[list_all[k], 8:9] + trackmate[list_all[k] + 1,10:11]

                loss.backward()
                optimizer.step()
                loss_list.append(loss.item()/regressive_step)


        if np.mean(loss_list) < best_loss:
            best_loss = np.mean(loss_list)
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', 'best_model_new.pt'))
            print(f"Epoch: {epoch} Loss: {np.round(np.mean(loss_list), 4)} saving model")
        else:
            print(f"Epoch: {epoch} Loss: {np.round(np.mean(loss_list), 4)}")

        list_plot.append(np.mean(loss_list))

        model.a.data = torch.clamp(model.a.data, min=-4, max=4)
        embedding = model.a.detach().cpu().numpy()

        for k in range(nDataset):
            embedding[k] = scaler.fit_transform(embedding[k])

        fig = plt.figure(figsize=(13, 8))
        # plt.ion()

        ax = fig.add_subplot(2, 3, 1)
        plt.plot(list_plot, color='k')
        if epoch<100:
            plt.xlim([0, 100])
        else:
            plt.xlim([0, 500])
        if list_plot[-1] < 1:
            plt.ylim([0, 1])
        elif list_plot[-1] < 10:
            plt.ylim([0, 10])
        else:
            plt.ylim([0, 20])
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Epochs', fontsize=10)

        for k in range(nDataset):
            ax = fig.add_subplot(2, 3, 4+k)
            plt.scatter(embedding[k][:, 0], embedding[k][:, 1], s=1, color='k')
            plt.xlabel('Embedding 0', fontsize=12)
            plt.ylabel('Embedding 1', fontsize=12)
            plt.xlim([-3.1, 3.1])
            plt.ylim([-3.1, 3.1])

        if (epoch % 10 == 0) & (epoch > 0):
            best_loss = np.mean(loss_list)
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', 'best_model_new.pt'))
            R2_list = test_model(model_config, trackmate_list=trackmate_list, bVisu=False, bMinimization=False, frame_start=160)
            model.train()

        if (epoch > 9):

            ax = fig.add_subplot(2, 3, 2)
            plt.plot(160+np.arange(len(R2_list)), R2_list, c='k')
            plt.ylim([0, 1])
            plt.xlabel('Frame', fontsize=12)
            plt.ylabel('R2', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"./tmp_training/Fig_{ntry}_{epoch}.tif")
        plt.close()


    print(' end of training')
    print(' ')

class MLP2(torch.nn.Module):
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

        self.upgrade_typeidden_size = hidden_size
        self.upgrade_type = h
        self.layers = layers
        self.msg = msg
        self.device =device

        if (self.msg == 0) | (self.msg==3):
             self.lin_edge = MLP2(input_size=1, hidden_size=self.upgrade_typeidden_size, output_size=7, layers=self.layers, device=self.device)
        if (self.msg == 1):
            self.lin_edge = MLP2(input_size=13, hidden_size=self.upgrade_typeidden_size, output_size=7, layers=self.layers, device=self.device)
        if (self.msg == 2):
            self.lin_edge = MLP2(input_size=14, hidden_size=self.upgrade_typeidden_size, output_size=7, layers=self.layers, device=self.device)

        if self.upgrade_type == 0:
            self.lin_node = MLP2(input_size=12, hidden_size=self.upgrade_typeidden_size, output_size=7, layers=self.layers, device=self.device)
        else:
            self.lin_node = MLP2(input_size=64, hidden_size=self.upgrade_typeidden_size, output_size=7, layers=self.layers, device=self.device)

        self.rnn = torch.nn.GRUCell(14, 64,device=self.device,dtype=torch.float64)

    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        if self.upgrade_type==0:
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

        self.upgrade_type = h
        self.layers = layers
        self.device =device
        self.embedding = embedding

        self.lin_edge = MLP2(input_size=3*self.embedding, hidden_size=3*self.embedding, output_size=self.embedding, layers= self.layers, device=self.device)

        if self.upgrade_type == 0:
            self.lin_node = MLP2(input_size=2*self.embedding, hidden_size=2*self.embedding, output_size=self.embedding, layers= self.layers, device=self.device)
        else:
            self.lin_node = MLP2(input_size=64, hidden_size=self.embedding, output_size=self.embedding, layers= self.layers, device=self.device)
            self.rnn = torch.nn.GRUCell(14, 64,device=self.device,dtype=torch.float64)

    def forward(self, x, edge_index, edge_feature):

        aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

        if self.upgrade_type==0:
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

    def __init__(self, model_config, device):
        super().__init__()

        self.upgrade_typeidden_size = model_config['hidden_size']
        self.embedding = model_config['embedding']
        self.nlayers = model_config['n_mp_layers']
        self.upgrade_type = model_config['upgrade_type']
        self.noise_level = model_config['noise_level']
        self.n_tracks = model_config['n_tracks']
        self.msg = model_config['msg']
        self.device = device
        self.output_angle = model_config['output_angle']
        self.cell_embedding = model_config['cell_embedding']
        self.edge_init = EdgeNetwork(self.upgrade_typeidden_size, layers=3)

        if self.embedding > 0:
            self.layer = torch.nn.ModuleList(
                [InteractionNetworkEmb(layers=3, embedding=self.embedding, h=self.upgrade_type, device=self.device) for _ in
                 range(self.nlayers)])
            self.node_out = MLP2(input_size=self.embedding, hidden_size=self.upgrade_typeidden_size, output_size=1, layers=3,
                                device=self.device)
        else:
            self.layer = torch.nn.ModuleList([InteractionNetwork(hidden_size=self.upgrade_typeidden_size, layers=3, h=self.upgrade_type,
                                                                 msg=self.msg, device=self.device) for _ in
                                              range(self.nlayers)])
            self.node_out = MLP2(input_size=7, hidden_size=self.upgrade_typeidden_size, output_size=1, layers=3,
                                device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((3, int(self.n_tracks + 1), self.cell_embedding)), requires_grad=False, device=self.device))
        self.t = nn.Parameter(
            torch.tensor(np.ones((3, int(self.n_tracks + 1), self.cell_embedding)), requires_grad=False, device=self.device))

        self.upgrade_type_all = torch.zeros((int(self.n_tracks + 1), 64), requires_grad=False, device=self.device,dtype=torch.float64)

        if self.embedding > 0:
            self.embedding_node = MLP2(input_size=5 + self.cell_embedding, hidden_size=self.embedding, output_size=self.embedding,
                                      layers=3, device=self.device)
            self.embedding_edges = MLP2(input_size=5, hidden_size=self.embedding, output_size=self.embedding,
                                       layers=3, device=self.device)

    def forward(self, data, data_id):

        node_feature = torch.cat((data.x[:, 2:6], data.x[:, 8:9]), dim=-1)
        noise = torch.randn((node_feature.shape[0], node_feature.shape[1]), requires_grad=False,
                            device=self.device) * self.noise_level
        node_feature = node_feature + noise
        edge_feature = self.edge_init(node_feature, data.edge_index, edge_feature=data.edge_attr)

        if self.embedding > 0:
            node_feature = self.embedding_node(torch.cat((node_feature, self.a[data_id, data.x[:, 1].detach().cpu().numpy(), :]), dim=-1))
            edge_feature = self.embedding_edges(edge_feature)

        for i in range(self.nlayers):
            node_feature, edge_feature = self.layer[i](node_feature, data.edge_index, edge_feature=edge_feature)

        pred = self.node_out(node_feature)

        return pred

def train_model_ResNet(model_config=None, trackmate_list=None, nstd=None, nmean=None, n_tracks_list=None):

    ntry = model_config['ntry']
    print('ntry: ', model_config['ntry'])
    if model_config['upgrade_type'] == 0:
        print('no GRUcell ')
    elif model_config['upgrade_type'] == 1:
        print('with GRUcell ')
    if (model_config['msg'] == 0) | (model_config['embedding'] > 0):
        print('msg: 0')
    elif model_config['msg'] == 1:
        print('msg: MLP2(x_jp, y_jp, vx_p, vy_p, ux, uy)')
    elif model_config['msg'] == 2:
        print('msg: MLP2(x_jp, y_jp, vx_p, vy_p, ux, uy, diff erkj)')
    elif model_config['msg'] == 3:
        print('msg: MLP2(diff_erk)')
    else:  # model_config['msg']==4:
        print('msg: 0')
    output_angle = model_config['output_angle']

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(ntry))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'data', 'val_outputs'), exist_ok=True)
    copyfile(os.path.realpath(__file__), os.path.join(log_dir, 'training_code.py'))

    model = ResNetGNN(model_config=model_config, device=device)
    # state_dict = torch.load(f"./log/try_{ntry}/models/best_model_new.pt")
    # model.load_state_dict(state_dict['model_state_dict'])

    print('model = ResNetGNN()   predict derivative Erk ')

    if model_config['train_MLPs'] == False:
        print('No MLP2s training watch out')
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
    model.train()

    print('Training model = ResNetGNN() ...')
    print(' ')

    data_id = 0
    trackmate = trackmate_list[data_id].copy()
    trackmate_true = trackmate_list[data_id].copy()

    best_loss = np.inf


    for epoch in range(1000):

        if epoch == 100:
            optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)  # , weight_decay=5e-3)

        mserr_list = []

        for data_id in range(len(model_config['dataset'])):
            trackmate = trackmate_list[data_id].copy()
            trackmate_true = trackmate_list[data_id].copy()

            for frame in range(20, model_config['frame_end'][data_id]):  # frame_list:

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
                pred = model(data=dataset, data_id=data_id)

                loss = criteria((pred[:, :] + x[:, 8:9]) * mask, target * mask) * 3

                loss.backward()
                optimizer.step()
                mserr_list.append(loss.item())

                trackmate[list_all + 1, 10:11] = np.array(pred.detach().cpu())
                trackmate[list_all + 1, 8:9] = trackmate[list_all, 8:9] + trackmate[list_all + 1, 10:11]

            print(f"data_id: {data_id} Epoch: {epoch} Loss: {np.round(np.mean(mserr_list), 4)}")

            if (np.mean(mserr_list) < best_loss) & (data_id==0)  :
                best_loss = np.mean(mserr_list)
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', 'best_model_new.pt'))

def test_model(model_config=None, trackmate_list=None, bVisu=False, bMinimization=False, frame_start=20):

    print('')
    ntry = model_config['ntry']
    dataset = model_config['dataset']
    file_folder = model_config['file_folder']
    dx = model_config['dx']
    dt = model_config['dt']
    metric_list = model_config['metric_list']
    frame_end = model_config['frame_end']
    net_type = model_config['net_type']
    time_window = model_config['time_window']

    model_lin = LinearRegression()

    ntry = model_config['ntry']

    if net_type == 'InteractionParticles':
        model = InteractionParticles(model_config=model_config, device=device)
        model.nstd = nstd
        model.nmean = nmean
        state_dict = torch.load(f"./log/try_{ntry}/models/best_model_new.pt")
        model.load_state_dict(state_dict['model_state_dict'])

    if net_type == 'ResNetGNN':
        model = ResNetGNN(model_config=model_config, device=device)
        model.nstd = nstd
        model.nmean = nmean
        state_dict = torch.load(f"./log/try_{ntry}/models/best_model_new.pt")
        model.load_state_dict(state_dict['model_state_dict'])


    print('UMAP embedding...')

    reducer = umap.UMAP()

    pos = torch.argwhere(torch.sum(model.a[0,:, :],axis=1) != 8)
    pos = pos[:, 0].detach().cpu().numpy()
    coeff_norm = model.a[0,pos.astype(int)].data.detach().cpu().numpy()

    # coeff_df = pd.DataFrame(coeff, columns=['Column_A', 'Column_B', 'Column_C','Column_D','Column_E', 'Column_F', 'Column_G','Column_H'])
    # sns.pairplot(coeff_df)
    # plt.show()

    trans = umap.UMAP(n_neighbors=10, n_components=2, random_state=42, transform_queue_size=0).fit(coeff_norm)
    # v = trans.transform(coeff_norm)
    # print(v[10])
    # print(trans.transform(coeff_norm[10:11, :]))

    print(f' UMAP done ...')
    # plt.ion()
    # plt.scatter(
    #     v[:, 0],
    #     v[:, 1], s=10, color='k')
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP', fontsize=18);
    # plt.show()


    criteria = nn.MSELoss()
    model.eval()

    rmserr_list = []

    true_ERK1 = []
    model_ERK1 = []
    true_ERK2 = []
    model_ERK2 = []
    drift_list = []

    R2model = []
    R2c = []
    R2area = []
    R2narea = []
    R2speed = []

    data_id = 0
    trackmate = trackmate_list[data_id].copy()
    trackmate_true = trackmate_list[data_id].copy()

    message_track = np.zeros((trackmate.shape[0], 2))
    I = imread(f'{file_folder}/../ACTIVITY.tif')
    I = np.array(I)

    frame_start

    for frame in tqdm(range(frame_start, 240)):  # frame_list:

        model.frame = int(frame)
        pos = np.argwhere(trackmate[:, 0] == frame)

        list_all = pos[:, 0].astype(int)
        mask = torch.tensor(np.ones(list_all.shape[0]), device=device)
        for k in range(len(mask)):
            if trackmate[list_all[k] - time_window, 1] != trackmate[list_all[k] + 1, 1]:
                mask[k] = 0
            if torch.sum(model.a[0, trackmate[list_all[k], 1].astype(int), :]) == model_config['cell_embedding']:
                mask[k] = 0
            if (trackmate[list_all[k], 2] < -0.45) | (trackmate[list_all[k], 3] < -0.5) | (trackmate[list_all[k], 3] > 0.52):
               mask[k] = 0
        mask = mask[:, None]

        x = torch.tensor(trackmate[list_all, 0:17], device=device)
        if time_window > 0:
            for k in range(time_window):
                x = torch.cat((x, torch.tensor(trackmate_list[data_id][list_all - k -1, 8:9], device=device)), axis=-1)

        target = torch.tensor(trackmate_true[list_all + 1, 8:9], device=device)
        target_pos = torch.tensor(trackmate_true[list_all, 2:4], device=device)

        dataset = data.Data(x=x, pos=x[:, 2:4])
        transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
        dataset = transform(dataset)
        distance = dataset.edge_attr.detach().cpu().numpy()
        pos = np.argwhere(distance < model_config['radius'])
        edges = dataset.edge_index
        dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                            edge_attr=torch.tensor(distance[pos[:, 0]], device=device))

        pred = model(data=dataset, data_id=0, step=2, cos_phi=0, sin_phi=0)

        loss = criteria((pred[:, :] + x[:, 8:9]) * mask * nstd[8], target * mask * nstd[8])

        for k in range(len(mask)):
            if mask[k] == 1:
                trackmate[list_all[k] + 1, 10:11] = np.array(pred[k].detach().cpu())
                trackmate[list_all[k] + 1, 8:9] = trackmate[list_all[k], 8:9] + trackmate[list_all[k] + 1, 10:11]

        rmserr_list.append(np.sqrt(loss.item()))
        xx = target.detach().cpu().numpy()
        yy = pred + x[:, 8:9]
        yy = yy.detach().cpu().numpy()
        model_lin.fit(xx, yy)
        R2model.append(model_lin.score(xx, yy))

        # trackmate[list_all + 1,4:5] = yy

        pos = np.argwhere(trackmate[list_all, 1] == 100)
        if len(pos) > 0:
            true_ERK1.append(trackmate_true[list_all[pos[0]] + 1, 8])
            model_ERK1.append(trackmate[list_all[pos[0]] + 1, 8])
        pos = np.argwhere(trackmate[list_all, 1] == 200)
        if len(pos) > 0:
            true_ERK2.append(trackmate_true[list_all[pos[0]] + 1, 8])
            model_ERK2.append(trackmate[list_all[pos[0]] + 1, 8])

        if bVisu & (frame%4==0): #frame == 200:

            # print(f'{frame} {np.round(loss.item(), 3)}  {np.round(model_lin.score(xx, yy), 3)} mask {np.round(torch.sum(mask).item() / mask.shape[0], 3)}')

            fig = plt.figure(figsize=(30, 18))
            plt.ion()

            ax = fig.add_subplot(3, 5, 11)
            v = trans.transform(coeff_norm)
            plt.scatter(v[:, 0],v[:, 1], s=1, color=[0.75,0.75,0.75])
            # plt.gca().set_aspect('equal', 'datalim')
            pos = torch.argwhere(mask==1)
            pos = pos[:,0].detach().cpu().numpy()
            current_coeff = model.a[0,trackmate_true[list_all[pos.astype(int)], 1],:].detach().cpu().numpy()
            v = trans.transform(current_coeff)
            plt.scatter(v[:, 0],v[:, 1], s=5, color='k')
            plt.xlabel('UMAP-0 [a.u]', fontsize=12)
            plt.ylabel('UMAP-1 [a.u]', fontsize=12)

            ax = fig.add_subplot(3, 5, 1)
            plt.scatter(trackmate_true[list_all + 1, 2], trackmate_true[list_all + 1, 3], s=125, marker='.',
                        c=target.detach().cpu().numpy(), vmin=-0.6, vmax=0.6)
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])
            plt.text(-0.6, 0.7, 'True ERK', fontsize=12)

            ax = fig.add_subplot(3, 5, 10)
            plt.scatter(trackmate_true[list_all[pos.astype(int)] + 1, 2], trackmate_true[list_all[pos.astype(int)] + 1, 3], s=25, marker='.',color=[0.75,0.75,0.75])
            pos = torch.argwhere(mask==0)
            pos = pos[:,0].detach().cpu().numpy()
            plt.scatter(trackmate_true[list_all[pos.astype(int)] + 1, 2],
                        trackmate_true[list_all[pos.astype(int)] + 1, 3], s=25, marker='.', color='k')
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])

            ax = fig.add_subplot(3, 5, 2)
            plt.scatter(trackmate[list_all + 1, 2], trackmate[list_all + 1, 3], s=125, marker='.',
                        c=(x[:, 8:9] + pred).detach().cpu().numpy(), vmin=-0.6, vmax=0.6)
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])
            plt.text(-0.6, 0.7, 'Model ERK', fontsize=12)

            ax = fig.add_subplot(3, 5, 3)
            xx0 = x.detach().cpu()
            xx0[:, 0] = xx0[:, 2]
            xx0[:, 1] = xx0[:, 3]
            xx0 = xx0[:, 0:2]
            pos = np.argwhere(trackmate[list_all + 1, 1] == 100)

            dataset = data.Data(x=x, pos=xx0)
            transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
            dataset = transform(dataset)
            distance = dataset.edge_attr.detach().cpu().numpy()
            pos = np.argwhere(distance < model_config['radius'])
            edges = dataset.edge_index
            dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                                edge_attr=torch.tensor(distance[pos[:, 0]], device=device))
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            # nx.draw(vis, pos=npos, node_size=10, linewidths=0)
            npos = dict(enumerate(np.array(xx0), 0))
            nx.draw_networkx(vis, pos=npos, node_size=0, linewidths=0, with_labels=False, edge_color='k')
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])

            ax = fig.add_subplot(3, 5, 5)
            model_lin = LinearRegression()

            plt.plot(np.arange(20, 20 + len(R2model)), np.array(R2model), 'k', label='Model')
            plt.xlim([0, 240])
            plt.ylim([-0.1, 1.1])
            plt.xlabel('Frame [a.u.]', fontsize=12)
            plt.ylabel('R2 coeff. [a.u.]', fontsize=12)
            xx = target.detach().cpu().numpy()
            yy = trackmate[list_all, 14:15]
            model_lin.fit(xx, yy)
            R2c.append(model_lin.score(xx, yy))
            plt.plot(20+4*np.arange(len(R2c)), np.array(R2c), 'b', label='Cell density')
            yy = trackmate[list_all, 12:13]
            model_lin.fit(xx, yy)
            R2area.append(model_lin.score(xx, yy))
            plt.plot(20+4*np.arange(len(R2area)), np.array(R2area), 'g', label='Cell area')
            yy = np.sqrt(trackmate[list_all, 4:5] ** 2 + trackmate[list_all, 5:6] ** 2)
            model_lin.fit(xx, yy)
            R2speed.append(model_lin.score(xx, yy))
            plt.plot(20+4*np.arange(len(R2speed)), np.array(R2speed), 'c', label='Cell velocity')
            plt.legend(loc='upper left', fontsize=12)

            ax = fig.add_subplot(3, 5, 4)
            pos = torch.argwhere(mask==1)
            pos = pos[:,0].detach().cpu().numpy()
            plt.scatter(trackmate_true[list_all[pos.astype(int)] + 1, 8:9], trackmate[list_all[pos.astype(int)] + 1, 8:9], s=1, c='k')
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            xx = target.detach().cpu().numpy()
            yy = (x[:, 8:9] + pred).detach().cpu().numpy()
            model_lin.fit(xx, yy)
            plt.text(-1, 1.3,
                     f"R2: {np.round(model_lin.score(xx, yy), 3)}   slope: {np.round(model_lin.coef_[0][0], 2)}   N: {pos.shape[0]} / {xx.shape[0]}",
                     fontsize=12)
            plt.xlabel('True ERk [a.u]', fontsize=12)
            plt.ylabel('Model Erk [a.u]', fontsize=12)

            # if net_type == 'InteractionParticles':
            #     ax = fig.add_subplot(3, 5, 11)
            #     plt.scatter(trackmate[list_all + 1, 2], trackmate[list_all + 1, 3], s=125, marker='.',
            #                 c=message[:, 0], vmin=-1, vmax=1)
            #     plt.xlim([-0.6, 0.95])
            #     plt.ylim([-0.6, 0.6])
            #     plt.text(-0.6, 0.65, 'Message amplitude [a.u.]', fontsize=12)

            ax = fig.add_subplot(3, 5, 6)
            xx0 = x.detach().cpu() * nstd[2]
            xx0[:, 0] = xx0[:, 2] + nmean[2]
            xx0[:, 1] = xx0[:, 3] + nmean[3]
            xx0 = xx0[:, 0:2] / dx
            pos = np.argwhere(trackmate[list_all + 1, 1] == 100)
            if len(pos) > 0:
                cx = xx0[pos[0], 0]
                cy = xx0[pos[0], 1]
            t = np.squeeze(I[frame, :, :])
            plt.imshow(t, vmax=2)
            dataset = data.Data(x=x, pos=xx0)
            transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
            dataset = transform(dataset)
            distance = dataset.edge_attr.detach().cpu().numpy()
            pos = np.argwhere(distance < model_config['radius'] * nstd[2])
            edges = dataset.edge_index
            dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                                edge_attr=torch.tensor(distance[pos[:, 0]], device=device))
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            # nx.draw(vis, pos=npos, node_size=10, linewidths=0)
            npos = dict(enumerate(np.array(xx0), 0))
            nx.draw_networkx(vis, pos=npos, node_size=0, linewidths=0, with_labels=False, edge_color='w')
            plt.xlim([cx - 50, cx + 50])
            plt.ylim([cy - 50, cy + 50])
            plt.text(cx - 50, cy + 52, 'Track 100', fontsize=12)

            e = edges.detach().cpu().numpy()
            pos = np.argwhere(trackmate[list_all, 1] == 100)
            if len(pos) > 0:
                ppos = np.argwhere(e[0, :] == pos[0])
                e = np.squeeze(xx0[e[1, ppos], :])
                e = e.detach().cpu().numpy()
                points = np.concatenate((e, xx0[pos[0], :].detach().cpu().numpy()))
                vor = Voronoi(points)
                regions, vertices = voronoi_finite_polygons_2d(vor)
                pts = MultiPoint([Point(i) for i in points])
                mask = pts.convex_hull
                new_vertices = []
                for region in regions:
                    polygon = vertices[region]
                    shape = list(polygon.shape)
                    shape[0] += 1
                    p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
                    poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
                    new_vertices.append(poly)
                plt.fill(*zip(*poly), c='w', alpha=0.5)

                for k in list_all:
                    plt.arrow(x=(trackmate[k, 2] * nstd[2] + nmean[2])/dx, y=(trackmate[k, 3] * nstd[2] + nmean[3])/dx,
                              dx=trackmate[k, 24] * nstd[4] * 2 * dt, dy=trackmate[k, 25] * nstd[5] * 2 * dt, head_width=2,
                              length_includes_head=True)
                    plt.arrow(x=(trackmate[k, 2] * nstd[2] + nmean[2])/dx - trackmate[k, 4] * nstd[4] * 2 * dt,
                              y=(trackmate[k, 3] * nstd[2] + nmean[3])/dx - trackmate[k, 5] * nstd[5] * 2 * dt,
                              dx=trackmate[k, 4] * nstd[4] * 2 * dt, dy=trackmate[k, 5] * nstd[5] * 2 * dt, head_width=2,
                              alpha=0.5, length_includes_head=True)

            ax = fig.add_subplot(3, 5, 7)
            plt.plot(np.arange(20, 20 + len(true_ERK1)), np.array(true_ERK1) * nstd[8] + nmean[8], 'g',
                     label='True ERK')
            plt.plot(np.arange(20, 20 + len(model_ERK1)), np.array(model_ERK1) * nstd[8] + nmean[8], 'k',
                     label='Model ERK')
            plt.ylim([1, 2])
            plt.xlim([0, 240])
            plt.legend(loc='upper right')
            plt.xlabel('Frame [a.u]', fontsize=12)

            # ax = fig.add_subplot(3, 5, 12)
            # pos = np.argwhere(trackmate[:, 1] == 100)
            # ppos = np.argwhere((trackmate[pos, 0] < frame + 1) & (trackmate[pos, 0] > 19))
            # plt.plot(trackmate[pos[ppos, 0], 0], trackmate[pos[ppos, 0], 12], 'y',
            #          label='Norm. Area')
            # plt.plot(trackmate[pos[ppos, 0], 0], trackmate[pos[ppos, 0], 13], 'g',
            #          label='Norm. Perimeter')
            # plt.plot(trackmate[pos[ppos, 0], 0],
            #          np.sqrt(trackmate[pos[ppos, 0], 4] ** 2 + trackmate[pos[ppos, 0], 5] ** 2), 'm',
            #          label='Norm. velocity')
            # plt.ylim([-1, 1])
            # plt.xlim([0, 240])
            # plt.xlabel('Frame [a.u]', fontsize=12)
            # plt.legend()
            # handles, labels = ax.get_legend_handles_labels()
            # unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            # ax.legend(*zip(*unique))

            ax = fig.add_subplot(3, 5, 8)
            pos = np.argwhere(trackmate[list_all, 1] == 200)
            if len(pos) > 0:
                c2x = xx0[pos[0], 0]
                c2y = xx0[pos[0], 1]
            t = np.squeeze(I[frame, :, :])
            plt.imshow(t, vmax=2)
            dataset = data.Data(x=x, pos=xx0)
            transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
            dataset = transform(dataset)
            distance = dataset.edge_attr.detach().cpu().numpy()
            pos = np.argwhere(distance < model_config['radius'] * nstd[2])
            edges = dataset.edge_index
            dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                                edge_attr=torch.tensor(distance[pos[:, 0]], device=device))
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            # nx.draw(vis, pos=npos, node_size=10, linewidths=0)
            npos = dict(enumerate(np.array(xx0), 0))
            nx.draw_networkx(vis, pos=npos, node_size=0, linewidths=0, with_labels=False, edge_color='w')
            plt.xlim([c2x - 50, c2x + 50])
            plt.ylim([c2y - 50, c2y + 50])
            plt.text(c2x - 50, c2y + 52, 'Track 200', fontsize=12)

            e = edges.detach().cpu().numpy()
            pos = np.argwhere(trackmate[list_all, 1] == 200)
            if len(pos) > 0:
                ppos = np.argwhere(e[0, :] == pos[0])
                e = np.squeeze(xx0[e[1, ppos], :])
                e = e.detach().cpu().numpy()
                points = np.concatenate((e, xx0[pos[0], :].detach().cpu().numpy()))
                vor = Voronoi(points)

                regions, vertices = voronoi_finite_polygons_2d(vor)

                pts = MultiPoint([Point(i) for i in points])
                mask = pts.convex_hull
                new_vertices = []
                for region in regions:
                    polygon = vertices[region]
                    shape = list(polygon.shape)
                    shape[0] += 1
                    p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
                    poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
                    new_vertices.append(poly)
                plt.fill(*zip(*poly), c='w', alpha=0.5)
                for k in list_all:
                        plt.arrow(x=(trackmate[k, 2] * nstd[2] + nmean[2]) / dx,
                                  y=(trackmate[k, 3] * nstd[2] + nmean[3]) / dx,
                                  dx=trackmate[k, 24] * nstd[4] * 2 * dt, dy=trackmate[k, 25] * nstd[5] * 2 * dt,
                                  head_width=2,
                                  length_includes_head=True)
                        plt.arrow(x=(trackmate[k, 2] * nstd[2] + nmean[2]) / dx - trackmate[k, 4] * nstd[4] * 2 * dt,
                                  y=(trackmate[k, 3] * nstd[2] + nmean[3]) / dx - trackmate[k, 5] * nstd[5] * 2 * dt,
                                  dx=trackmate[k, 4] * nstd[4] * 2 * dt, dy=trackmate[k, 5] * nstd[5] * 2 * dt,
                                  head_width=2,
                                  alpha=0.5, length_includes_head=True)

            ax = fig.add_subplot(3, 5, 9)
            plt.plot(np.arange(20, 20 + len(true_ERK2)), np.array(true_ERK2) * nstd[8] + nmean[8], 'g',
                     label='True ERK')
            plt.plot(np.arange(20, 20 + len(model_ERK2)), np.array(model_ERK2) * nstd[8] + nmean[8], 'k',
                     label='Model ERK')
            plt.ylim([1, 2])
            plt.xlim([0, 240])
            plt.legend(loc='upper right')
            plt.xlabel('Frame [a.u]', fontsize=12)

            # ax = fig.add_subplot(3, 5, 14)
            # pos = np.argwhere(trackmate[:, 1] == 200)
            # ppos = np.argwhere((trackmate[pos, 0] < frame + 1) & (trackmate[pos, 0] > 19))
            # plt.plot(trackmate[pos[ppos, 0], 0], trackmate[pos[ppos, 0], 12], 'y',
            #          label='Norm. Area')
            # plt.plot(trackmate[pos[ppos, 0], 0], trackmate[pos[ppos, 0], 13], 'g',
            #          label='Norm. Perimeter')
            # plt.plot(trackmate[pos[ppos, 0], 0],
            #          np.sqrt(trackmate[pos[ppos, 0], 4] ** 2 + trackmate[pos[ppos, 5], 7] ** 2), 'm',
            #          label='Norm. velocity')
            # plt.ylim([-1, 1])
            # plt.xlim([0, 240])
            # plt.xlabel('Frame [a.u]', fontsize=12)
            # plt.legend()
            # handles, labels = ax.get_legend_handles_labels()
            # unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            # ax.legend(*zip(*unique))

            # plt.show()

            plt.savefig(f"./tmp_recons/Fig_{ntry}_{frame}.tif")
            plt.close()

    # print(f"RMSE: {np.round(np.mean(rmserr_list), 3)} +/- {np.round(np.std(rmserr_list), 3)}     {np.round(np.mean(rmserr_list) / nstd[4] * 3, 1)} sigma ")
    # print(f"Erk: {np.round(nmean[8], 3)} +/- {np.round(nstd[8] / 3, 3)} ")
    print(f"R2: {np.round(np.mean(R2model), 3)} +/- {np.round(np.std(R2model), 3)} ")
    print('')

    return R2model

def load_model_config (id=505):

    model_config_test=[]

    if id==505:
        model_config_test = {'ntry': 505,
                    'dataset': ['2309012_490'],
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
                    'upgrade_type': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'noise_level': 0,
                    'batch_size': 4,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200], # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 3,
                    'net_type':'InteractionParticles'}
    if id == 506:
        model_config_test = {'ntry': 506,
                    'dataset': ['2309012_490', '2309012_491', '2309012_492'],
                    'trackmate_metric': {'Label': 0,
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
                    'metric_list': ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch1', 'Area'],
                    'file_folder': '/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/',
                    'dx': 0.908,
                    'dt': 5.0,
                    'upgrade_type': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode': 1,
                    'time_embedding': False,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'noise_level': 0,
                    'batch_size': 4,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 3,
                    'net_type': 'InteractionParticles'}
    if id == 507:
        model_config_test = {'ntry': 507,
                    'dataset': ['2309012_490'],
                    'trackmate_metric': {'Label': 0,
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
                    'metric_list': ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch1', 'Area'],
                    'file_folder': '/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/',
                    'dx': 0.908,
                    'dt': 5.0,
                    'upgrade_type': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode': 1,
                    'time_embedding': False,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'noise_level': 0,
                    'batch_size': 4,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 8,
                    'net_type': 'InteractionParticles'}
    if id == 508:
        model_config_test = {'ntry': 508,
                    'dataset': ['2309012_490'],
                    'trackmate_metric': {'Label': 0,
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
                    'metric_list': ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch1', 'Area'],
                    'file_folder': '/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/',
                    'dx': 0.908,
                    'dt': 5.0,
                    'upgrade_type': 2,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode': 1,
                    'time_embedding': False,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'noise_level': 0,
                    'batch_size': 4,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 3,
                    'net_type': 'InteractionParticles'}
    if id == 509:
        model_config_test = {'ntry': 509,
                    'dataset': ['2309012_490'],
                    'trackmate_metric': {'Label': 0,
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
                    'metric_list': ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch1', 'Area'],
                    'file_folder': '/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/',
                    'dx': 0.908,
                    'dt': 5.0,
                    'upgrade_type': 3,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode': 1,
                    'time_embedding': False,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'noise_level': 0,
                    'batch_size': 4,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 3,
                    'net_type': 'InteractionParticles'}
    if id == 510:
        model_config_test = {'ntry': 510,
                    'dataset': '2309012_490',
                    'trackmate_metric': {'Label': 0,
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
                    'metric_list': ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch1', 'Area'],
                    'file_folder': '/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/',
                    'dx': 0.908,
                    'dt': 5.0,
                    'upgrade_type': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode': 1,
                    'embedding': 128,
                    'time_embedding': False,
                    'n_mp_layers': 5,
                    'hidden_size': 32,
                    'noise_level': 0,
                    'batch_size': 8,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 3,
                    'net_type': 'ResNetGNN'}
    if id == 511:
        model_config_test = {'ntry': 511,
                    'dataset': ['2309012_490', '2309012_491', '2309012_492'],
                    'trackmate_metric': {'Label': 0,
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
                    'metric_list': ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch1', 'Area'],
                    'file_folder': '/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/',
                    'dx': 0.908,
                    'dt': 5.0,
                    'upgrade_type': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode': 1,
                    'embedding': 128,
                    'time_embedding': False,
                    'n_mp_layers': 5,
                    'hidden_size': 32,
                    'noise_level': 0,
                    'batch_size': 8,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 3,
                    'net_type': 'ResNetGNN'}
    if id == 512:
        model_config_test = {'ntry': 512,
                    'dataset':  ['2309012_490'], #  ['2309012_490', '2309012_491', '2309012_492'],
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
                    'upgrade_type': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers0': 5,
                    'hidden_size0': 64,
                    'output_size0': 6,
                    'n_mp_layers1':2,
                    'hidden_size1': 16,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'noise_level': 0,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200],   #   [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 2,
                    'batch_size':8,
                    'net_type':'InteractionParticles',
                    'training_mode': 't+1'}
    if id == 513:
        model_config_test = {'ntry': 513,
                         'dataset': ['2309012_490'],  # ['2309012_490', '2309012_491', '2309012_492'],
                         'trackmate_metric': {'Label': 0,
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
                         'metric_list': ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch1', 'Area'],
                         'file_folder': '/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/',
                         'dx': 0.908,
                         'dt': 5.0,
                         'upgrade_type': 0,
                         'msg': 1,
                         'aggr': 0,
                         'rot_mode': 1,
                         'time_embedding': False,
                         'n_mp_layers0': 5,
                         'hidden_size0': 64,
                         'output_size0': 1,
                         'n_mp_layers1': 2,
                         'hidden_size1': 16,
                         'n_mp_layers': 5,
                         'hidden_size': 128,
                         'bNoise': False,
                         'noise_level': 0,
                         'batch_size': 4,
                         'time_window': 0,
                         'frame_start': 20,
                         'frame_end': [200],  # [200, 182, 182],  # [241,228,228],
                         'n_tracks': 0,
                         'radius': 0.15,
                         'remove_update_U': True,
                         'train_MLPs': True,
                         'cell_embedding': 2,
                         'batch_size': 4,
                         'net_type': 'InteractionParticles',
                         'training_mode': 't+1'}
    if id == 514:
        model_config_test = {'ntry': 514,
                    'dataset':  ['2309012_490', '2309012_491', '2309012_492'],
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
                    'upgrade_type': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers0': 5,
                    'hidden_size0': 64,
                    'output_size0': 6,
                    'n_mp_layers1':2,
                    'hidden_size1': 16,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'noise_level': 0,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 2,
                    'batch_size':8,
                    'net_type':'InteractionParticles',
                    'training_mode': 't+1'}
    if id == 515:
        model_config_test = {'ntry': 515,
                    'dataset':  ['2309012_490', '2309012_491', '2309012_492'],
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
                    'upgrade_type': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers0': 5,
                    'hidden_size0': 64,
                    'output_size0': 1,
                    'n_mp_layers1':2,
                    'hidden_size1': 16,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'noise_level': 0,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 2,
                    'batch_size':8,
                    'net_type':'InteractionParticles',
                    'training_mode': 't+1'}
    if id == 516:
        model_config_test = {'ntry': 512,
                    'dataset':  ['2309012_490'], #  ['2309012_490', '2309012_491', '2309012_492'],
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
                    'upgrade_type': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers0': 5,
                    'hidden_size0': 128,
                    'output_size0': 6,
                    'n_mp_layers1':2,
                    'hidden_size1': 16,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'noise_level': 0,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200],   #   [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 2,
                    'batch_size':8,
                    'net_type':'InteractionParticles',
                    'training_mode': 't+1'}
    if id == 517:
        model_config_test = {'ntry': 513,
                         'dataset': ['2309012_490'],  # ['2309012_490', '2309012_491', '2309012_492'],
                         'trackmate_metric': {'Label': 0,
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
                         'metric_list': ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch1', 'Area'],
                         'file_folder': '/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/',
                         'dx': 0.908,
                         'dt': 5.0,
                         'upgrade_type': 0,
                         'msg': 1,
                         'aggr': 0,
                         'rot_mode': 1,
                         'time_embedding': False,
                         'n_mp_layers0': 5,
                         'hidden_size0': 128,
                         'output_size0': 1,
                         'n_mp_layers1': 2,
                         'hidden_size1': 16,
                         'n_mp_layers': 5,
                         'hidden_size': 128,
                         'bNoise': False,
                         'noise_level': 0,
                         'time_window': 0,
                         'frame_start': 20,
                         'frame_end': [200],  # [200, 182, 182],  # [241,228,228],
                         'n_tracks': 0,
                         'radius': 0.15,
                         'remove_update_U': True,
                         'train_MLPs': True,
                         'cell_embedding': 2,
                         'batch_size': 8,
                         'net_type': 'InteractionParticles',
                         'training_mode': 't+1'}
    if id == 518:
        model_config_test = {'ntry': 518,
                    'dataset':  ['2309012_490'], #  ['2309012_490', '2309012_491', '2309012_492'],
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
                    'upgrade_type': 0,
                    'msg': 2,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers0': 5,
                    'hidden_size0': 128,
                    'output_size0': 16,
                    'n_mp_layers1':2,
                    'hidden_size1': 16,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200],   #   [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 2,
                    'batch_size':8,
                    'net_type':'InteractionParticles',
                    'training_mode': 't+1'}
    if id == 519:
        model_config_test = {'ntry': 519,
                    'dataset':  ['2309012_490'], #  ['2309012_490', '2309012_491', '2309012_492'],
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
                    'upgrade_type': 0,
                    'msg': 2,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers0': 5,
                    'hidden_size0': 128,
                    'output_size0': 16,
                    'n_mp_layers1':2,
                    'hidden_size1': 16,
                    'n_mp_layers': 5,
                    'hidden_size': 128,
                    'noise_level': 0,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200],   #   [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 2,
                    'batch_size':8,
                    'net_type':'InteractionParticles',
                    'training_mode': 'regressive'}
    if id == 520:
        model_config_test = {'ntry': 520,
                    'dataset':  ['2309012_490'], #  ['2309012_490', '2309012_491', '2309012_492'],
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
                    'upgrade_type': 0,
                    'msg': 2,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers0': 5,
                    'hidden_size0': 64,
                    'output_size0': 16,
                    'n_mp_layers1':2,
                    'hidden_size1': 16,
                    'n_mp_layers': 5,
                    'noise_level': 0,
                    'time_window': 0,
                    'frame_start': 20,
                    'frame_end': [200],   #   [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'remove_update_U': True,
                    'train_MLPs': True,
                    'cell_embedding': 2,
                    'batch_size':8,
                    'net_type':'InteractionParticles',
                    'training_mode': 't+1'}
    if id == 521:
        model_config_test = {'ntry': 521,
                    'dataset':  ['2309012_490'], #  ['2309012_490', '2309012_491', '2309012_492'],
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
                    'upgrade_type': 0,
                    'msg': 2,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers0': 5,
                    'hidden_size0': 64,
                    'output_size0': 16,
                    'n_mp_layers1': 3,
                    'hidden_size1': 16,
                    'noise_level': 0,
                    'time_window': 4,
                    'frame_start': 20,
                    'frame_end': [200],   #   [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'cell_embedding': 2,
                    'batch_size':8,
                    'net_type':'InteractionParticles',
                    'training_mode': 't+1'}
    if id == 522:
        model_config_test = {'ntry': 522,
                    'dataset':  ['2309012_490'], #  ['2309012_490', '2309012_491', '2309012_492'],
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
                    'upgrade_type': 0,
                    'msg': 2,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers0': 5,
                    'hidden_size0': 64,
                    'output_size0': 16,
                    'n_mp_layers1': 3,
                    'hidden_size1': 16,
                    'noise_level': 0,
                    'time_window': 2,
                    'frame_start': 20,
                    'frame_end': [200],   #   [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'cell_embedding': 2,
                    'batch_size':8,
                    'net_type':'InteractionParticles',
                    'training_mode': 't+1'}
    if id == 523:
        model_config_test = {'ntry': 523,
                    'dataset':  ['2309012_490'], #  ['2309012_490', '2309012_491', '2309012_492'],
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
                    'upgrade_type': 0,
                    'msg': 2,
                    'aggr': 0,
                    'rot_mode':1,
                    'time_embedding': False,
                    'n_mp_layers0': 5,
                    'hidden_size0': 64,
                    'output_size0': 16,
                    'n_mp_layers1': 3,
                    'hidden_size1': 16,
                    'noise_level': 5E-3,
                    'time_window': 2,
                    'frame_start': 20,
                    'frame_end': [200],   #   [200, 182, 182],  # [241,228,228],
                    'n_tracks': 0,
                    'radius': 0.15,
                    'cell_embedding': 2,
                    'batch_size':8,
                    'net_type':'InteractionParticles',
                    'training_mode': 't+1'}

    # print('watch out model_config not find')
    return model_config_test


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Version 1.0 24 sept 2023')

    print(f'Device :{device}')

    scaler = StandardScaler()
    S_e = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    gtest_list = [519, 521, 522]

    for gtest in range(522,523):

        model_config = load_model_config(id=gtest)
        for key, value in model_config.items():
            print(key, ":", value)

        if model_config['msg'] == 0:
            print('msg: 0')
        elif model_config['msg'] == 1:
            print('msg: MLP(d, x_jp, y_jp, vx_ip, vy_ip, vx_jp, vy_jp, diff_erk)')
        elif model_config['msg'] == 2:
            print('msg: MLP(d, x_jp, y_jp, vx_ip, vy_ip, vx_jp, vy_jp, erkj)')
        if model_config['cell_embedding'] == 0:
            print('embedding: a false t false')
        elif model_config['cell_embedding'] == 1:
            print('embedding: a true t false')
        else:  # model_config['cell_embedding'] == 2:
            print('embedding: a true t true')

        trackmate_list, nstd, nmean, n_tracks_list = load_trackmate(model_config)
        # train_model_Interaction(model_config, trackmate_list, nstd, nmean, n_tracks_list)
        R2_list = test_model(model_config, trackmate_list=trackmate_list, bVisu=True, bMinimization=False, frame_start=20)
        # train_model_ResNet(model_config, trackmate_list, nstd, nmean, n_tracks_list)












