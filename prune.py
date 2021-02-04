import numpy as np
import torch


def prune_weights(torchweights, prune_coeff):
    weights = np.abs(torchweights.cpu().numpy());
    weightshape = weights.shape
    rankedweights = weights.reshape(weights.size).argsort()
    
    num = weights.size
    prune_num = int(np.round(num*prune_coeff))
    count = 0
    masks = np.zeros_like(rankedweights)
    for n, rankedweight in enumerate(rankedweights):
        if rankedweight > prune_num:
            masks[n] = 1
        else: 
            count += 1
    #print("total weights:", num)
    #print("weights pruned:", count)
    
    masks = masks.reshape(weightshape)
    weights = masks*weights
    
    return torch.from_numpy(weights).cuda(), masks


def ReadAllWeights(net):
    parameters = net.state_dict()
    all_layers = list(net.state_dict().keys())
    modules = []
    for i in range(len(all_layers)):
        modules.append(all_layers[i].split('.')[:-1])

    weights_grouped_by_layers = []
    layer_name = []
    module_name = []
    for layer in range(len(all_layers)):
        levels = modules[layer]
        current_layer = net
        for k in range(len(levels)):
            current_layer = current_layer._modules.get(levels[k])
        if 'weight' in all_layers[layer]:
            weights = current_layer.weight.data.cpu().numpy().flatten()
            print(current_layer, ':  \t\t\t',len(weights))
            weights_grouped_by_layers.append(weights)
            layer_name.append(all_layers[layer])
            module_name.append(current_layer)
    return module_name, layer_name, weights_grouped_by_layers