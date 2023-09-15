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
from graph_HGF_ERK_dim128 import *



def test_model(bVisu=False):

    ff = 0

    file_list=["/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210105/",\
               "/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210108/",\
               "/home/allierc@hhmi.org/Desktop/signaling/HGF-ERK signaling/fig 1/B_E/210109/"]
    file_folder = file_list[ff]
    print(file_folder)


    model_lin = LinearRegression()

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
    if model_config['cell_embedding'] == 0:
        print('embedding: a false t false')
    elif model_config['cell_embedding'] == 1:
        print('embedding: a true t false')
    else:  # model_config['cell_embedding']=2
        print('embedding: a true t true')


    trackmate = trackmate_list[ff].copy()
    trackmate_true = trackmate_list[ff].copy()

    model = ResNetGNN(model_config=model_config, device=device)
    model.nstd = nstd
    model.nmean = nmean
    state_dict = torch.load(f"./log/try_{ntry}/models/best_model_new_emb_concatenate.pt")
    model.load_state_dict(state_dict['model_state_dict'])

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
    I = imread(f'{file_folder}/ACTIVITY.tif')
    I = np.array(I)

    # pos = np.argwhere(trackmate[:, 1] > 20)
    # trackmate[pos, 4:5] = 0
    # trackmate[pos, 8:9] = 0

    for frame in tqdm(range(175, 240)):  # frame_list:

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

        x = torch.tensor(trackmate[list_all, 0:19], device=device)
        prev_x = x.clone()

        target = torch.tensor(trackmate_true[list_all + 1, 4:5], device=device)

        dataset = data.Data(x=x, pos=x[:, 2:4])
        transform = T.Compose([T.Delaunay(), T.FaceToEdge(), T.Distance(norm=False)])
        dataset = transform(dataset)
        distance = dataset.edge_attr.detach().cpu().numpy()
        pos = np.argwhere(distance < model_config['radius'])
        edges = dataset.edge_index
        dataset = data.Data(x=x, edge_index=edges[:, pos[:, 0]],
                            edge_attr=torch.tensor(distance[pos[:, 0]], device=device))

        pred = model(data=dataset, data_id=ff)

        loss = criteria((pred[:, :] + x[:, 4:5]) * mask, target * mask) * 3

        if True:   # Erk is upodatge according to the predicted derivative of Erk

            for k in range(len(mask)):
                if mask[k] == 1:
                    trackmate[list_all + 1, 8:9] = np.array(pred.detach().cpu())
                    trackmate[list_all + 1, 4:5] = trackmate[list_all, 4:5] + trackmate[list_all + 1, 8:9]

        else:  # by how much one must drift the cells to get the right derivative of Erk at next timestep0

            target_ERK_deriv = torch.tensor(trackmate_true[list_all + 1, 8:9], device=device)

            erk_move = Erk_move(GNN=model, x=x, data=dataset, data_id=ff, device=device)
            optimizer = torch.optim.Adam(erk_move.parameters(), lr=1E-5)

            for loop in range(200):
                pred = erk_move()
                loss = criteria(pred * mask * nstd[4], target_ERK_deriv * mask * nstd[4])
                loss.backward()
                optimizer.step()

            for k in range(len(mask)):
                if mask[k] == 1:
                    # trackmate[list_all[k]+1, 2:4] = erk_move.new_x[k,:].detach().cpu().numpy()                                                            # by how much cells must move ?
                    # trackmate[list_all[k]+1, 6:8] = (trackmate[list_all[k]+1, 2:4] - trackmate[list_all[k], 2:4]) * nstd[2]/nstd[6]
                    # trackmate[list_all[k]+1, 2:4] = trackmate[list_all[k], 2:4] + erk_move.new_x[k,:].detach().cpu().numpy() *nstd[6]/nstd[2]               # by how much cells must change speed ?
                    # trackmate[list_all[k]+1, 6:8] = erk_move.new_x[k, :].detach().cpu().numpy()
                    trackmate[list_all[k]+1, 2:4] = 1/3 * erk_move.new_x[k,2:4].detach().cpu().numpy()  +  2/3 * (trackmate[list_all[k], 2:4] + erk_move.new_x[k, 6:8].detach().cpu().numpy() * nstd[6] / nstd[2])       # combination of the two
                    trackmate[list_all[k] + 1, 6:8] = (trackmate[list_all[k] + 1, 2:4] - trackmate[list_all[k], 2:4]) * nstd[2] / nstd[6]

        rmserr_list.append(np.sqrt(loss.item()))
        xx = target.detach().cpu().numpy()
        yy = pred + x[:, 4:5]
        yy = yy.detach().cpu().numpy()
        model_lin.fit(xx, yy)
        R2model.append(model_lin.score(xx, yy))

        # trackmate[list_all + 1,4:5] = yy

        pos = np.argwhere(trackmate[list_all, 1] == 100)
        if len(pos) > 0:
            true_ERK1.append(trackmate_true[list_all[pos[0]] + 1, 4])
            model_ERK1.append(trackmate[list_all[pos[0]] + 1, 4])
        pos = np.argwhere(trackmate[list_all, 1] == 200)
        if len(pos) > 0:
            true_ERK2.append(trackmate_true[list_all[pos[0]] + 1, 4])
            model_ERK2.append(trackmate[list_all[pos[0]] + 1, 4])



        if bVisu:  # frame == 200:

            fig = plt.figure(figsize=(18, 18))
            #plt.ion()

            ax = fig.add_subplot(3, 3, 1)
            plt.scatter(trackmate_true[list_all + 1, 2], trackmate_true[list_all + 1, 3], s=125, marker='.',
                        c=target.detach().cpu().numpy(), vmin=-0.6, vmax=0.6)
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])
            plt.text(-0.6, 0.7, 'True ERK', fontsize=12)

            ax = fig.add_subplot(3, 3, 2)
            plt.scatter(trackmate[list_all + 1, 2], trackmate[list_all + 1, 3], s=125, marker='.',
                        c=(x[:, 4:5] + pred).detach().cpu().numpy(), vmin=-0.6, vmax=0.6)
            plt.xlim([-0.6, 0.95])
            plt.ylim([-0.6, 0.6])
            plt.text(-0.6, 0.7, 'Model ERK', fontsize=12)

            ax = fig.add_subplot(3, 3, 3)
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

            ax = fig.add_subplot(3, 3, 4)
            plt.scatter(trackmate_true[list_all + 1, 4:5], trackmate[list_all + 1, 4:5], s=1, c='k')
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            xx = target.detach().cpu().numpy()
            yy = (x[:, 4:5] + pred).detach().cpu().numpy()
            model_lin.fit(xx, yy)
            plt.text(-1, 0.9,
                     f"R2: {np.round(model_lin.score(xx, yy), 3)}   slope: {np.round(model_lin.coef_[0][0], 1)}   N: {xx.shape[0]}",
                     fontsize=12)
            plt.xlabel('True ERk [a.u]', fontsize=12)
            plt.ylabel('Model Erk [a.u]', fontsize=12)

            # ax = fig.add_subplot(3, 5, 11)
            # plt.scatter(trackmate[list_all + 1, 2], trackmate[list_all + 1, 3], s=125, marker='.',
            #             c=message[:, 0], vmin=-1, vmax=1)
            # plt.xlim([-0.6, 0.95])
            # plt.ylim([-0.6, 0.6])
            # plt.text(-0.6, 0.65, 'Message amplitude [a.u.]', fontsize=12)

            ax = fig.add_subplot(3, 3, 5)
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
            plt.legend(loc='upper right', fontsize=8)

            ax = fig.add_subplot(3, 3, 7)
            xx0 = x.detach().cpu() * nstd[2]
            xx0[:, 0] = xx0[:, 2] + nmean[2]
            xx0[:, 1] = xx0[:, 3] + nmean[3]
            xx0 = xx0[:, 0:2]
            pos = np.argwhere(trackmate[list_all + 1, 1] == 100)
            if len(pos) > 0:
                cx = xx0[pos[0], 0]
                cy = xx0[pos[0], 1]
            t = np.squeeze(I[frame, :, :])
            plt.imshow(t, vmin=0, vmax=2)
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
                    plt.arrow(x=trackmate[k, 2] * nstd[2] + nmean[2], y=trackmate[k, 3] * nstd[2] + nmean[3],
                              dx=trackmate[k, 46] * nstd[6], dy=trackmate[k, 47] * nstd[6], head_width=2,
                              length_includes_head=True)
                    plt.arrow(x=trackmate[k, 2] * nstd[2] + nmean[2] - trackmate[k, 6] * nstd[6] * 2,
                              y=trackmate[k, 3] * nstd[2] + nmean[3] - trackmate[k, 7] * nstd[6] * 2,
                              dx=trackmate[k, 6] * nstd[6] * 2, dy=trackmate[k, 7] * nstd[6] * 2, head_width=2,
                              alpha=0.5, length_includes_head=True)

            ax = fig.add_subplot(3, 3, 8)
            plt.plot(np.arange(20, 20 + len(true_ERK1)), np.array(true_ERK1) * nstd[4] + nmean[4], 'g',
                     label='True ERK')
            plt.plot(np.arange(20, 20 + len(model_ERK1)), np.array(model_ERK1) * nstd[4] + nmean[4], 'k',
                     label='Model ERK')
            plt.ylim([1, 2])
            plt.xlim([0, 240])
            plt.legend(loc='upper right')
            plt.xlabel('Frame [a.u]', fontsize=12)

            ax = fig.add_subplot(3, 3, 9)
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



            # plt.show()

            plt.savefig(f"./ReconsGraph/Fig_{frame}.tif")
            plt.close()

    print(f"RMSE: {np.round(np.mean(rmserr_list), 3)} +/- {np.round(np.std(rmserr_list), 3)}     {np.round(np.mean(rmserr_list) / nstd[4] * 3, 1)} sigma ")
    print(f"Erk: {np.round(nmean[4], 3)} +/- {np.round(nstd[4] / 3, 3)} ")
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


    model_config = {'ntry': 420,
                    'h': 0,
                    'msg': 1,
                    'embedding': 128,
                    'cell_embedding': 1,
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

    test_model(bVisu=False)

    # model_config = {'ntry': 422,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'embedding': 64,
    #                 'cell_embedding': 1,
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
    # test_model(bVisu=False)
    #
    #
    # model_config = {'ntry': 423,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'embedding': 128,
    #                 'cell_embedding': 1,
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
    #
    # test_model(bVisu=False)

    # model_config = {'ntry': 424,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'embedding': 128,
    #                 'cell_embedding': 0,
    #                 'output_angle': False,
    #                 'n_mp_layers': 5,
    #                 'hidden_size': 32,
    #                 'noise_level': 5E-3,
    #                 'batch_size': 8,
    #                 'bRollout': True,
    #                 'rollout_window': 2,
    #                 'remove_update_U': True,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    #
    # test_model(bVisu=False)

    # model_config = {'ntry': 425,
    #                 'h': 0,
    #                 'msg': 1,
    #                 'embedding': 128,
    #                 'cell_embedding': 1,
    #                 'output_angle': False,
    #                 'n_mp_layers': 5,
    #                 'hidden_size': 32,
    #                 'noise_level': 5E-3,
    #                 'batch_size': 8,
    #                 'bRollout': True,
    #                 'rollout_window': 2,
    #                 'remove_update_U': True,
    #                 'frame_start': 20,
    #                 'frame_end': [241, 228, 228],
    #                 'n_tracks': 3561,
    #                 'radius': 0.15}
    #
    # test_model(bVisu=False)

