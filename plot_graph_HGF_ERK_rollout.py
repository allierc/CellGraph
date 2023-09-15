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
from graph_HGF_ERK_rollout import *


def test_model(bVisu=False, bMinimization=False):


    model_lin = LinearRegression()

    ntry = model_config['ntry']
    bRollout = model_config['bRollout']

    print(f'Testing {ntry}')
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
        print('msg: MLP(diff_erk)')
    else:  # model_config['msg']==4:
        print('msg: 0')
    if model_config['cell_embedding'] == 0:
        print('embedding: a false t false')
    elif model_config['cell_embedding'] == 1:
        print('embedding: a true t false')
    else:  # model_config['cell_embedding'] == 2:
        print('embedding: a true t true')


    trackmate_true = trackmate.copy()

    model = InteractionParticlesRollout(model_config=model_config, device=device)
    model.nstd = nstd
    model.nmean = nmean
    state_dict = torch.load(f"./log/try_{ntry}/models/best_model_new.pt")
    model.load_state_dict(state_dict['model_state_dict'])

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

    message_track = np.zeros((trackmate.shape[0], 2))
    I = imread(f'{file_folder}/../ACTIVITY.tif')
    I = np.array(I)

    for frame in tqdm(range(20, 240)):  # frame_list:

        model.frame = int(frame)
        pos = np.argwhere(trackmate[:, 0] == frame)

        list_all = pos[:, 0].astype(int)
        mask = torch.tensor(np.ones(list_all.shape[0]), device=device)
        for k in range(len(mask)):
            if trackmate[list_all[k] - 1, 1] != trackmate[list_all[k] + 1, 1]:
                mask[k] = 0
            # if (trackmate[list_all[k], 2] < -0.35) | (trackmate[list_all[k], 3] < -0.45) | (
            #         trackmate[list_all[k], 3] > 0.45):
            #    mask[k] = 0
        mask = mask[:, None]

        x = torch.tensor(trackmate[list_all, 0:17], device=device)

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

        message, pred = model(data=dataset, data_id=0)

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
        message = message.detach().cpu().numpy()

        # trackmate[list_all + 1,4:5] = yy

        pos = np.argwhere(trackmate[list_all, 1] == 100)
        if len(pos) > 0:
            true_ERK1.append(trackmate_true[list_all[pos[0]] + 1, 8])
            model_ERK1.append(trackmate[list_all[pos[0]] + 1, 8])
        pos = np.argwhere(trackmate[list_all, 1] == 200)
        if len(pos) > 0:
            true_ERK2.append(trackmate_true[list_all[pos[0]] + 1, 8])
            model_ERK2.append(trackmate[list_all[pos[0]] + 1, 8])

        if bVisu: #frame == 200:

            # print(f'{frame} {np.round(loss.item(), 3)}  {np.round(model_lin.score(xx, yy), 3)} mask {np.round(torch.sum(mask).item() / mask.shape[0], 3)}')

            fig = plt.figure(figsize=(32, 18))
            #plt.ion()

            ax = fig.add_subplot(3, 5, 1)
            plt.scatter(trackmate_true[list_all + 1, 2], trackmate_true[list_all + 1, 3], s=125, marker='.',
                        c=target.detach().cpu().numpy(), vmin=-0.6, vmax=0.6)
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])
            plt.text(-0.6, 0.7, 'True ERK', fontsize=12)

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

            ax = fig.add_subplot(3, 5, 4)
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
            plt.plot(np.arange(20, 20 + len(R2c)), np.array(R2c), 'b', label='Cell density')
            yy = trackmate[list_all, 12:13]
            model_lin.fit(xx, yy)
            R2area.append(model_lin.score(xx, yy))
            plt.plot(np.arange(20, 20 + len(R2area)), np.array(R2area), 'g', label='Cell area')
            yy = trackmate[list_all, 8:9]
            model_lin.fit(xx, yy)
            R2narea.append(model_lin.score(xx, yy))
            plt.plot(np.arange(20, 20 + len(R2narea)), np.array(R2narea), 'r', label='Signal 2')
            yy = np.sqrt(trackmate[list_all, 4:5] ** 2 + trackmate[list_all, 5:6] ** 2)
            model_lin.fit(xx, yy)
            R2speed.append(model_lin.score(xx, yy))
            plt.plot(np.arange(20, 20 + len(R2speed)), np.array(R2speed), 'c', label='Cell velocity')
            plt.legend(loc='upper left', fontsize=12)

            ax = fig.add_subplot(3, 5, 5)
            plt.scatter(trackmate_true[list_all + 1, 8:9], trackmate[list_all + 1, 8:9], s=1, c='k')
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            xx = target.detach().cpu().numpy()
            yy = (x[:, 8:9] + pred).detach().cpu().numpy()
            model_lin.fit(xx, yy)
            plt.text(-1, 1.3,
                     f"R2: {np.round(model_lin.score(xx, yy), 3)}   slope: {np.round(model_lin.coef_[0][0], 2)}   N: {xx.shape[0]}",
                     fontsize=12)
            plt.xlabel('True ERk [a.u]', fontsize=12)
            plt.ylabel('Model Erk [a.u]', fontsize=12)

            ax = fig.add_subplot(3, 5, 11)
            plt.scatter(trackmate[list_all + 1, 2], trackmate[list_all + 1, 3], s=125, marker='.',
                        c=message[:, 0], vmin=-1, vmax=1)
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])
            plt.text(-0.6, 0.65, 'Message amplitude [a.u.]', fontsize=12)

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

            ax = fig.add_subplot(3, 5, 12)
            pos = np.argwhere(trackmate[:, 1] == 100)
            ppos = np.argwhere((trackmate[pos, 0] < frame + 1) & (trackmate[pos, 0] > 19))
            plt.plot(trackmate[pos[ppos, 0], 0], trackmate[pos[ppos, 0], 11], 'y',
                     label='Norm. Area')
            plt.plot(trackmate[pos[ppos, 0], 0], trackmate[pos[ppos, 0], 12], 'g',
                     label='Norm. Perimeter')
            plt.plot(trackmate[pos[ppos, 0], 0],
                     np.sqrt(trackmate[pos[ppos, 0], 4] ** 2 + trackmate[pos[ppos, 0], 7] ** 2), 'm',
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

            # plt.show()

            plt.savefig(f"./tmp_recons/Fig_{frame}.tif")
            plt.close()

    print(f"RMSE: {np.round(np.mean(rmserr_list), 3)} +/- {np.round(np.std(rmserr_list), 3)}     {np.round(np.mean(rmserr_list) / nstd[4] * 3, 1)} sigma ")
    print(f"Erk: {np.round(nmean[8], 3)} +/- {np.round(nstd[8] / 3, 3)} ")
    print(f"R2: {np.round(np.mean(R2model), 3)} +/- {np.round(np.std(R2model), 3)} ")
    print('')

if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(f'Device :{device}')

    # model_config = {'ntry': 500,
    #                 'datum': '2309012_490',
    #                 'trackmate_metric' : {'Label': 0,
    #                 'Spot_ID': 1,
    #                 'Track_ID': 2,
    #                 'Quality': 3,
    #                 'X': 4,
    #                 'Y': 5,
    #                 'Z': 6,
    #                 'T': 7,
    #                 'Frame': 8,
    #                 'R': 9,
    #                 'Visibility': 10,
    #                 'Spot color': 11,
    #                 'Mean Ch1': 12,
    #                 'Median Ch1': 13,
    #                 'Min Ch1': 14,
    #                 'Max Ch1': 15,
    #                 'Sum Ch1': 16,
    #                 'Std Ch1': 17,
    #                 'Ctrst Ch1': 18,
    #                 'SNR Ch1': 19,
    #                 'El. x0': 20,
    #                 'El. y0': 21,
    #                 'El. long axis': 22,
    #                 'El. sh. axis': 23,
    #                 'El. angle': 24,
    #                 'El. a.r.': 25,
    #                 'Area': 26,
    #                 'Perim.': 27,
    #                 'Circ.': 28,
    #                 'Solidity': 29,
    #                 'Shape index': 30},
    #
    #                 'metric_list' : ['Frame', 'Track_ID', 'X', 'Y', 'Mean Ch1', 'Area'],
    #
    #                 'file_folder' : '/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/trackmate/',
    #
    #                 'dx':0.908,
    #                 'dt':5.0,
    #                 'h': 0,
    #                 'msg': 1,
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
    #                 'frame_end': [241],
    #                 'n_tracks': 0,
    #                 'radius': 0.15}

    model_config = {'ntry': 501,
                    'datum': '2309012_490',
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
                    'h': 0,
                    'msg': 1,
                    'aggr': 0,
                    'rot_mode': 1,
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



    test_model(bVisu=True, bMinimization=False)






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
    # test_model(bVisu=False, bMinimization=False)
    #
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
    # test_model(bVisu=False, bMinimization=False)

    # model_config = {'ntry': 373,
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
    # test_model(bVisu=False, bMinimization=False)
    #
    # model_config = {'ntry': 374,
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
    # test_model(bVisu=False, bMinimization=False)
    #
    # model_config = {'ntry': 375,
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
    # test_model(bVisu=False, bMinimization=False)
    #
    # model_config = {'ntry': 376,
    #                 'h': 0,
    #                 'msg': 4,
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
    # test_model(bVisu=False, minimization=False)











