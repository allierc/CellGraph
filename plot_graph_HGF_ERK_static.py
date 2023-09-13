
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
from sklearn.linear_model import LinearRegression
from shapely.geometry import MultiPoint, Point, Polygon
from graph_HGF_ERK_static import *

def test_model(bVisu=False):

    model_lin = LinearRegression()

    ntry=model_config['ntry']
    # bRollout = True

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

    model = InteractionParticles(model_config=model_config, device=device)
    # state_dict = torch.load(f"./log/try_{ntry}/models/best_model_new.pt")
    state_dict = torch.load(f"./log/try_{ntry}/models/best_model_new_emb_concatenate.pt")
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    criteria = nn.MSELoss()


    ff = 0
    trackmate = trackmate_list[ff].copy()
    trackmate_true = trackmate_list[ff].copy()

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

    message_track = np.zeros((trackmate.shape[0], 2))
    I = imread(f'{file_folder}/ACTIVITY.tif')
    I = np.array(I)

    pos = np.argwhere(trackmate[:, 0] > 21)
    trackmate[pos,4]=0
    trackmate[pos,8]=0


    for frame in tqdm(np.arange(20, 240)):

        pos = np.argwhere(trackmate[:, 0] == frame)

        list_all = pos[:,0].astype(int)
        mask=torch.tensor(np.ones(list_all.shape[0]), device=device)
        for k in range(len(mask)):
            if trackmate[list_all[k]-1,1]!=trackmate[list_all[k]+1,1]:
                mask[k]=0
        mask = mask[:, None]

        x = torch.tensor(trackmate[list_all, 0:14], device=device)
        prev_x=x

        dataset = data.Data(x=x, pos=x[:, 2:4])
        transform = T.Compose([T.Delaunay(),T.FaceToEdge(),T.Distance(norm=False)])
        dataset = transform(dataset)
        distance = dataset.edge_attr.detach().cpu().numpy()
        pos = np.argwhere(distance<model_config['radius'])
        edges = dataset.edge_index
        dataset = data.Data(x=x, edge_index=edges[:,pos[:,0]], edge_attr=torch.tensor(distance[pos[:, 0]],device=device))

        target = torch.tensor(trackmate_true[list_all, 4:5], device=device)
        target_ERK = torch.tensor(trackmate_true[list_all+1, 4:5], device=device)
        target_pos = torch.tensor(trackmate_true[list_all+1, 2:4], device=device)

        message, pred = model(data=dataset, data_id=ff)

        loss = criteria(pred * mask * nstd[4], target * mask * nstd[4])

        # x_= x.requires_grad_()
        # dataset = data.Data(x=x_, edge_index=edges[:,pos[:,0]], edge_attr=torch.tensor(distance[pos[:, 0]],device=device))
        # message, pred = model(data=dataset, data_id=ff)
        # energy = pred**2
        # drift = torch.autograd.grad(energy, x_, torch.ones_like(energy),create_graph = True)[0]

        if False:
            erk_move = Erk_move(GNN=model, x=x, data=dataset, data_id=ff, device=device)
            optimizer = torch.optim.Adam(erk_move.parameters(), lr=2.5E-4)
            for loop in range(1000):
                pred = erk_move()
                loss = criteria(pred * mask * nstd[4], target_ERK * mask * nstd[4])
                loss.backward()
                optimizer.step()
            for k in range(len(mask)):
                if mask[k] == 1:
                    # trackmate[list_all[k]+1, 2:4] = erk_move.new_x[k,2:4].detach().cpu().numpy()
                    # trackmate[list_all[k]+1, 6:8] = (trackmate[list_all[k]+1, 2:4] - trackmate[list_all[k], 2:4]) * nstd[2]/nstd[6]
                    trackmate[list_all[k]+1, 2:4] = trackmate[list_all[k], 2:4] + erk_move.new_x[k,6:8].detach().cpu().numpy() *nstd[6]/nstd[2]
                    trackmate[list_all[k]+1, 6:8] = erk_move.new_x[k, 6:8].detach().cpu().numpy()

        rmserr_list.append(np.sqrt(loss.item()))

        xx = target.detach().cpu().numpy()
        yy = pred.detach().cpu().numpy()
        model_lin.fit(xx, yy)
        R2model.append(model_lin.score(xx, yy))
        message = message.detach().cpu().numpy()

        cell_id=x[:, 1].detach().cpu().numpy()
        pos = np.argwhere(cell_id == 100)
        if len(pos) > 0:
            true_ERK1.append(np.squeeze(xx[pos[0]]))
            model_ERK1.append(np.squeeze(yy[pos[0]]))
        pos = np.argwhere(cell_id == 200)
        if len(pos) > 0:
            true_ERK2.append(np.squeeze(xx[pos[0]]))
            model_ERK2.append(np.squeeze(yy[pos[0]]))

        # if bRollout:
        #     trackmate[list_all + 1, 4:5] = np.array(pred.detach().cpu())
        #     # trackmate[list_all + 1, 4:5] = trackmate[list_all, 4:5] + trackmate[list_all + 1, 8:9]

        if bVisu:  # frame == 200:

            # print(
            #     f'{frame} {np.round(loss.item(), 3)}  {np.round(model_lin.score(xx, yy), 3)} mask {np.round(torch.sum(mask).item() / mask.shape[0], 3)}')

            fig = plt.figure(figsize=(32, 18))
            # plt.ion()

            ax = fig.add_subplot(3, 5, 1)
            plt.scatter(trackmate_true[list_all, 2], trackmate_true[list_all+1, 3], s=125, marker='.',
                        c=target.detach().cpu().numpy(), vmin=-0.6, vmax=0.6)
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])
            plt.text(-0.6, 0.7, 'True ERK', fontsize=12)

            ax = fig.add_subplot(3, 5, 2)
            plt.scatter(trackmate[list_all, 2], trackmate[list_all+1, 3], s=125, marker='.',
                        c=target.detach().cpu().numpy(), vmin=-0.6, vmax=0.6)
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])
            plt.text(-0.6, 0.7, 'Model ERK', fontsize=12)

            ax = fig.add_subplot(3, 5, 3)

            xx0 = x.detach().cpu()
            xx0[:, 0] = xx0[:, 2]
            xx0[:, 1] = xx0[:, 3]
            xx0 = xx0[:, 0:2]
            pos = np.argwhere(trackmate[list_all, 1] == 100)

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

            ax = fig.add_subplot(3, 5, 4)
            model_lin = LinearRegression()

            plt.plot(np.arange(20, 20 + len(R2model)), np.array(R2model), 'k', label='Model')
            plt.xlim([0, 240])
            plt.ylim([-0.1, 1.1])
            plt.xlabel('Frame [a.u.]', fontsize=12)
            plt.ylabel('R2 coeff. [a.u.]', fontsize=12)
            xx = target.detach().cpu().numpy()
            yy = trackmate[list_all, 13:14]
            model_lin.fit(xx, yy)
            R2c.append(model_lin.score(xx, yy))
            plt.plot(np.arange(20, 20 + len(R2c)), np.array(R2c), 'b', label='Cell density')
            yy = trackmate[list_all, 11:12]
            model_lin.fit(xx, yy)
            R2area.append(model_lin.score(xx, yy))
            plt.plot(np.arange(20, 20 + len(R2area)), np.array(R2area), 'g', label='Cell area')
            yy = trackmate[list_all, 5:6]
            model_lin.fit(xx, yy)
            R2narea.append(model_lin.score(xx, yy))
            plt.plot(np.arange(20, 20 + len(R2narea)), np.array(R2narea), 'r', label='Cell area')
            yy = np.sqrt(trackmate[list_all, 6:7] ** 2 + trackmate[list_all, 7:8] ** 2)
            model_lin.fit(xx, yy)
            R2speed.append(model_lin.score(xx, yy))
            plt.plot(np.arange(20, 20 + len(R2speed)), np.array(R2speed), 'c', label='Cell velocity')
            plt.legend(loc='upper left', fontsize=12)

            ax = fig.add_subplot(3, 5, 5)
            plt.scatter(trackmate_true[list_all, 4:5], pred.detach().cpu().numpy(), s=1, c='k')
            xx = target.detach().cpu().numpy()
            yy = pred.detach().cpu().numpy()
            model_lin.fit(xx, yy)
            plt.text(-1, 1.3,
                     f"R2: {np.round(model_lin.score(xx, yy), 3)}   slope: {np.round(model_lin.coef_[0][0], 2)}   N: {xx.shape[0]}",
                     fontsize=12)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.xlabel('True ERk [a.u]', fontsize=12)
            plt.ylabel('Model Erk [a.u]', fontsize=12)

            ax = fig.add_subplot(3, 5, 11)
            plt.scatter(trackmate[list_all, 2], trackmate[list_all, 3], s=125, marker='.',
                        c=message[:, 0], vmin=-2, vmax=2, )
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])
            plt.text(-0.6, 0.65, 'Message amplitude [a.u.]', fontsize=12)

            ax = fig.add_subplot(3, 5, 6)
            xx0 = x.detach().cpu() * nstd[2]
            xx0[:, 0] = xx0[:, 2] + nmean[2]
            xx0[:, 1] = xx0[:, 3] + nmean[3]
            xx0 = xx0[:, 0:2]
            pos = np.argwhere(trackmate[list_all, 1] == 100)
            if len(pos) > 0:
                cx = xx0[pos[0], 0]
                cy = xx0[pos[0], 1]
            t = np.squeeze(I[frame, :, :])
            plt.imshow(t, vmax=2)
            dataset = data.Data(x=x, pos=xx0)
            transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
            dataset = transform(dataset)
            distance = dataset.edge_attr.detach().cpu().numpy()
            pos = np.argwhere(distance<model_config['radius']/1.5)
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
                    plt.arrow(x=trackmate[k, 2] * nstd[2] + nmean[2], y=trackmate[k, 3] * nstd[2] + nmean[3],
                              dx=trackmate[k, 46] * nstd[6], dy=trackmate[k, 47] * nstd[6], head_width=2,
                              length_includes_head=True)
                    plt.arrow(x=trackmate[k, 2] * nstd[2] + nmean[2] - trackmate[k, 6] * nstd[6] * 2,
                              y=trackmate[k, 3] * nstd[2] + nmean[3] - trackmate[k, 7] * nstd[6] * 2,
                              dx=trackmate[k, 6] * nstd[6] * 2, dy=trackmate[k, 7] * nstd[6] * 2, head_width=2,
                              alpha=0.5, length_includes_head=True)

            ax = fig.add_subplot(3, 5, 7)
            plt.plot(np.arange(20, 20 + len(true_ERK1)), np.array(true_ERK1) * nstd[4] + nmean[4], 'g',
                     label='True ERK')
            plt.plot(np.arange(20, 20 + len(model_ERK1)), np.array(model_ERK1) * nstd[4] + nmean[4], 'k',
                     label='Model ERK')
            plt.ylim([1, 2])
            plt.xlim([0, 240])
            plt.legend(loc='upper right')
            plt.xlabel('Frame [a.u]', fontsize=12)

            ax = fig.add_subplot(3, 5, 12)
            pos = np.argwhere(trackmate[:, 1] == 100)
            ppos = np.argwhere((trackmate[pos, 0] < frame + 1) & (trackmate[pos, 0] > 19))
            plt.plot(trackmate[pos[ppos, 0], 0], trackmate[pos[ppos, 0], 11], 'y',
                     label='Norm. Area')
            plt.plot(trackmate[pos[ppos, 0], 0], trackmate[pos[ppos, 0], 12], 'g',
                     label='Norm. Perimeter')
            plt.plot(trackmate[pos[ppos, 0], 0],
                     np.sqrt(trackmate[pos[ppos, 0], 6] ** 2 + trackmate[pos[ppos, 0], 7] ** 2), 'm',
                     label='Norm. velocity')
            plt.ylim([-1, 1])
            plt.xlim([0, 240])
            plt.xlabel('Frame [a.u]', fontsize=12)
            plt.legend()
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            ax.legend(*zip(*unique))

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
            pos = np.argwhere(distance<model_config['radius']/1.5)
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
                plt.arrow(x=trackmate[k, 2] * nstd[2] + nmean[2], y=trackmate[k, 3] * nstd[2] + nmean[3],
                          dx=trackmate[k, 46] * nstd[6] * 2, dy=trackmate[k, 47] * nstd[6] * 2, head_width=2,
                          length_includes_head=True)
                plt.arrow(x=trackmate[k, 2] * nstd[2] + nmean[2] - trackmate[k, 6] * nstd[6] * 2,
                          y=trackmate[k, 3] * nstd[2] + nmean[3] - trackmate[k, 7] * nstd[6] * 2,
                          dx=trackmate[k, 6] * nstd[6] * 2, dy=trackmate[k, 7] * nstd[6] * 2, head_width=2,
                          alpha=0.5,
                          length_includes_head=True)

            ax = fig.add_subplot(3, 5, 9)
            plt.plot(np.arange(20, 20 + len(true_ERK2)), np.array(true_ERK2) * nstd[4] + nmean[4], 'g',
                     label='True ERK')
            plt.plot(np.arange(20, 20 + len(model_ERK2)), np.array(model_ERK2) * nstd[4] + nmean[4], 'k',
                     label='Model ERK')
            plt.ylim([1, 2])
            plt.xlim([0, 240])
            plt.legend(loc='upper right')
            plt.xlabel('Frame [a.u]', fontsize=12)

            ax = fig.add_subplot(3, 5, 14)
            pos = np.argwhere(trackmate[:, 1] == 200)
            ppos = np.argwhere((trackmate[pos, 0] < frame + 1) & (trackmate[pos, 0] > 19))
            plt.plot(trackmate[pos[ppos, 0], 0], trackmate[pos[ppos, 0], 11], 'y',
                     label='Norm. Area')
            plt.plot(trackmate[pos[ppos, 0], 0], trackmate[pos[ppos, 0], 12], 'g',
                     label='Norm. Perimeter')
            plt.plot(trackmate[pos[ppos, 0], 0],
                     np.sqrt(trackmate[pos[ppos, 0], 6] ** 2 + trackmate[pos[ppos, 0], 7] ** 2), 'm',
                     label='Norm. velocity')
            plt.ylim([-1, 1])
            plt.xlim([0, 240])
            plt.xlabel('Frame [a.u]', fontsize=12)
            plt.legend()
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            ax.legend(*zip(*unique))


            plt.savefig(f"./ReconsGraph/Fig_{frame}.tif")
            plt.close()

    print(f"RMSE: {np.round(np.mean(rmserr_list), 3)} +/- {np.round(np.std(rmserr_list), 3)}     {np.round(np.mean(rmserr_list)/nstd[4]*3, 1)} sigma ")
    print(f"Erk: {np.round(nmean[4], 3)} +/- {np.round(nstd[4]/3, 3)} ")
    print(f"R2: {np.round(np.mean(R2model), 3)} +/- {np.round(np.std(R2model), 3)} ")
    print('')

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Device :{device}')

    flist = ['ReconsGraph']
    for folder in flist:
        files = glob.glob(f"/home/allierc@hhmi.org/Desktop/Py/Graph/{folder}/*")
        for f in files:
            os.remove(f)

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

    model_config = {'ntry': 440,
                    'h': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode':1,
                    'embedding': 0,
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
    test_model(bVisu=False)

    model_config = {'ntry': 441,
                    'h': 0,
                    'msg': 2,
                    'aggr': 0,
                    'rot_mode':1,
                    'embedding': 0,
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
    test_model(bVisu=False)

    model_config = {'ntry': 442,
                    'h': 0,
                    'msg': 3,
                    'aggr': 0,
                    'rot_mode':1,
                    'embedding': 0,
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
    test_model(bVisu=False)
    #
    # model_config = {'ntry': 343,
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
    # test_model(bVisu=False)
    #
    # model_config = {'ntry': 344,
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
    # test_model(bVisu=False)
    #
    # model_config = {'ntry': 345,
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
    # test_model(bVisu=False)
    #
    # model_config = {'ntry': 346,
    #                 'h': 1,
    #                 'msg': 5,
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
    # test_model(bVisu=False)
    #
    # model_config = {'ntry': 347,
    #                 'h': 2,
    #                 'msg': 0,
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
    # test_model(bVisu=False)
    #
    # model_config = {'ntry': 348,
    #                 'h': 3,
    #                 'msg': 0,
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
    # test_model(bVisu=False)
    #
    # model_config = {'ntry': 349,
    #                 'h': 4,
    #                 'msg': 0,
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
    # test_model(bVisu=False)
    #
    # model_config = {'ntry': 350,
    #                 'h': 5,
    #                 'msg': 0,
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
    # test_model(bVisu=False)

    # x_ = x.requires_grad_()
    # dataset = data.Data(x=x_, edge_index=edges[:, pos[:, 0]],
    #                     edge_attr=torch.tensor(distance[pos[:, 0]], device=device))
    # message, pred = model(data=dataset, data_id=ff)
    # loss = criteria(pred * mask * nstd[4], target_ERK * mask * nstd[4])
    # drift = torch.autograd.grad(loss, x_, torch.ones_like(loss), create_graph=True)[0]
    #
    # x = x - 0.1*drift
    # x = x.detach()
    #
    # dataset = data.Data(x=x, pos=x[:, 2:4])
    # transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
    # dataset = transform(dataset)
    # distance = dataset.edge_attr.detach().cpu().numpy()
    # pos = np.argwhere(distance < model_config['radius'])
    # edges = dataset.edge_index
    # dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
    #                     edge_attr=torch.tensor(distance[pos[:, 0]], device=device))
    #
    # message, pred = model(data=dataset, data_id=ff)
    #
    # loss = criteria((pred[:, :] + x[:, 4:5]) * mask * nstd[4], target * mask * nstd[4])
    # loss2 = torch.sqrt(criteria((x[:, 2:4]) * mask, target_pos * mask)) * nstd[2]
    #
    # print(loss.item(), loss2.item() )











