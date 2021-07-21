import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
import json
from utils import *
from model import EdgeGNN

def gen_feat(nnode):
    types = np.ones(n_tot, dtype=int)
    slices = np.ones(n_tot, dtype=int)
    x = np.random.randint(32, size=n_tot)
    y = np.random.randint(32, size=n_tot)

    feat = np.array([types, slices, x, y]).T
    return feat

def loc_to_x_y(grid, loc):
    width, length = grid
    x = (loc % width)
    y = int(loc/width)
    return x, y 

def locs_to_coords(grid, locs_data):
    x_data, y_data = [], []
    
    for locs in locs_data:
        x_arr, y_arr = [], []
        for loc in locs:
            x, y = loc_to_x_y(grid, loc)
            x_arr.append(x)
            y_arr.append(y)
        x_data.append(x_arr)
        y_data.append(y_arr)

    return (x_data, y_data)

def gen_feat(num_data, num_node, coords):
    types = np.ones(num_node, dtype=int)
    slices = np.ones(num_node, dtype=int)

    feat = []
    for i in range(num_data):
        x = coords[0][i]
        y = coords[1][i]
        h = np.array([types, slices, x, y]).T
        feat.append(h)
    feat = np.array(feat)

    return feat

def prepare_dataset(data):
    num_node = data['nnode']
    grid = data['grid']
    edges = data['edges']
    plcs = data['placements']

    num_edge = len(edges[0])
    num_data = len(plcs)
    
    graph = dgl.graph((torch.tensor(edges[0]), torch.tensor(edges[1])))

    cost_labels = []
    locs = []
    for d in plcs.items():
        cost_labels.append(d[1]['cost']/num_edge)
        # cost_labels.append(d[1]['cost'])
        locs.append(d[1]['locs']) 

    coords = locs_to_coords(grid, locs)
    feat = gen_feat(num_data, num_node, coords)

    return (num_data, grid, graph, cost_labels, locs, feat)

def train(num_epoch, lr, num_layer, num_feat, num_hidden, data):
    model = EdgeGNN(num_layer, num_feat, num_hidden, F.relu)

    num_data, grid, g, cost_labels, locs, feat = prepare_dataset(data)

    g = dgl.add_reverse_edges(g)
    g = dgl.add_self_loop(g)
    
    labels = torch.tensor(cost_labels, dtype=torch.float)
    feat = torch.tensor(feat, dtype=torch.float)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    min_mean_loss = 99999999
    for e in range(num_epoch):
        with torch.autograd.set_detect_anomaly(True):
            loss_arr = []
            for i in range(num_data):
                model.train()
                p = model(g, feat[i])
                loss = loss_fn(p[0], labels[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_arr.append(loss.item())
                # print("Epoch: {:d}, Iteration: {:d}, Predict: {:.3f}, Label: {:.3f}, MSE Loss: {:.3f}".format(e, i, p.item(), labels[i].item(), loss))

            mean_loss = np.mean(np.array(loss_arr))
            print("Epoch: {:d}, Mean MSE Loss: {:.3f}".format(e, mean_loss))
            if mean_loss < min_mean_loss:
                min_mean_loss = mean_loss

    print("Min Mean MSE Loss: {:.3f}".format(min_mean_loss))

    return

if __name__ == '__main__':
    num_epoch = 100
    num_layer = 7
    num_feat = 4
    num_hidden = 32
    lr = 0.00001

    path = "./programs/mac"
    prog = "mac4"

    f = open('./dataset/' + prog + '_dataset.json',)
    data = json.load(f)

    train(num_epoch, lr, num_layer, num_feat, num_hidden, data)
