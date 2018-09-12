from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#for images
#IDEA: calculate R from X and Y but also add  R=d(x,y), 1, x/n, y/n, y%n, y%n
#add polar coordinates not only r but also O (the angle)
#allow some basic recursivety by pushing values through the network (up to n times and/or fixed max cycles per node)
#IDEA, maybe do a low res CPPN and upscale with conv and mutated kernels

import torch
import copy
from numpy.random import normal

from opensimplex import OpenSimplex

use_cuda = torch.cuda.is_available()
device="cpu"
if use_cuda:
    device="cuda"

def _tnsr(x,dtype=torch.float):
    return torch.tensor(x,device=device,dtype=dtype)
pi=_tnsr(np.pi)
e=_tnsr(np.e)

output_funs=[lambda x: 0.5 * torch.sin(x) + 0.5, lambda x: torch.sqrt(1.0-torch.abs(torch.tanh(x)))]
#TODO add all fractal flame functions, see electric sheep paper
#output from original paper: abs(tanh), sigm, gauss
class Fun(nn.Module):

    funs={"sin":torch.sin,"clam":lambda x : torch.clamp(x,-1,1),"cube":lambda x :torch.pow(x,3),
          "exp":torch.exp,"gaus": lambda x : 1/torch.sqrt(2*pi)*torch.pow(e,(-1/2)*torch.pow(x,2)),
          "hat": lambda x: nn.ReLU()(-torch.abs(x)+1), "sigm": torch.sigmoid,
          "softplus": nn.Softplus(), "square":lambda x : torch.pow(x,2), "tanh": torch.tanh,
          "id": lambda x :x}

    @staticmethod
    def get_n_rand(n):
        return [Fun(fun) for fun in np.random.choice(list(Fun.funs.keys()),n,replace=False)]
    #fun need to be a torch function e.g. torch.sin() or torch.abs()
    def __init__(self,fun):
        self.name=fun
        super(Fun, self).__init__()
        if fun is "rnd":
            self.fun = self.funs[np.random.choice(list(self.funs.keys()))]
        else:
            self.fun=self.funs[fun]
    def forward(self, x):
        return self.fun(x)

    def __str__(self):
        return self.name
import pickle
class CPPN(nn.Module):
    def file_extension(self):
        pass
    def cross_over(self):
        pass
    def mutate(self):
        pass
    def forward(self):
        pass
    def render_image(self,z,width,height,batch_size=500):
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)

        xv, yv = np.meshgrid(x, y)
        rv = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(xv, yv)]
        X, Y, R = np.reshape(xv, newshape=(-1, 1)), np.reshape(yv, newshape=(-1, 1)), np.reshape(rv, newshape=(-1, 1))
        Z = z.repeat(X.shape[0]).view(Y.shape[0], -1)
        X = _tnsr(X).to("cpu")
        Y = _tnsr(Y).to("cpu")
        R = _tnsr(R).to("cpu")
        Z = _tnsr(Z).to("cpu")

        image_out = []
        if batch_size==-1:
            batch_size=X.size(0)
        split = lambda x: torch.split(x, batch_size)
        for x, y, r,z in zip(*map(split, [X, Y, R,Z])):
            output = self.forward(x.to(device), y.to(device), r.to(device), z.to(device))
            output = output.data.to("cpu")
            image_out.append(output)
        image = np.reshape(torch.cat(image_out).data.numpy(), newshape=(width, height, self.output_size))
        return image
import dill
import datetime
class FC_CPPN(CPPN):
    def file_extension(self):
        return "fccppn"
    #maybe also allow full deserialisation
    #or find a way to init model without parameters and load from file

    def save(self,file_path,fe="fccppn",dated=True):
        start_time = str(datetime.datetime.now()) if dated else ""
        fe = "." + fe if fe is not "" else ""
        pickle_out = open(file_path+start_time+fe, "wb")
        dill.dump(self, pickle_out)
        pickle_out.close()
    @staticmethod
    def load(file_path,fe="fccppn"):
        fe = "." + fe if fe is not "" else ""
        pickle_in = open(file_path + fe, "rb")
        instance = dill.load(pickle_in)
        pickle_in.close()
        return instance
    def __init__(self,motion_size,fun_names,output_size=3,z_scale=10,n_layers=3,hidden_size=8):
        super(FC_CPPN, self).__init__()
        self.output_size=output_size
        self.z_scale=z_scale
        H = hidden_size
        self.fun_names=fun_names
        self.n_layers = n_layers
        #TODO also allow network without hidden layer
        #+3 because x,y,r is also an input
        self.FCz = nn.Sequential(nn.Linear(motion_size+3, H))
        self.middle_layers = nn.ModuleList([nn.Linear(H, H)] * n_layers)
        self.out_layer = nn.Sequential(
            nn.Linear(H, output_size),
            nn.Sigmoid()
        )
        self.Funs=nn.ModuleList([Fun(name) for name in fun_names])
        #group nodes that have the same activation function
        self.layer_masks=[]
        #TODO increase sparsness of function mask by adding deterministic dropout
        for layer in range(n_layers):
            cell_idxs = np.random.permutation(np.arange(H))
            # split the idxs seperate for each function
            layer_mask= [np.sort(arr) for arr in  np.array_split(cell_idxs, len(self.Funs))]
        #keep list of parameters
        #convert masks to torch Variables (with gradient?)
        #TODO if at some a gradient is desired the mask should be float instead of long
            #weighing all the different function activations
            self.layer_masks.extend([torch.nn.Parameter(_tnsr(mask,torch.long),requires_grad=False)
                                                      for mask in layer_mask])
        self.layer_masks=nn.ParameterList(self.layer_masks)

    def random_init_weights(self):
        zero_weigths_fract = 0.8
        weight_scale = 1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weight=normal(loc=0,scale=weight_scale,size=m.weight.data.size())
                # random boolean mask for which values will be changed
                mask = np.random.uniform(size=weight.shape)
                mask = np.around(mask*zero_weigths_fract)
                m.weight.data=_tnsr(weight*mask)
                bias= normal(loc=0, scale=weight_scale, size=m.bias.data.size())
                mask = np.random.uniform(size=bias.shape)
                mask = np.around(mask * zero_weigths_fract)
                m.bias.data=_tnsr(bias*mask)
    def forward(self, x,y,r,z):
        z=z/self.z_scale
        input = torch.cat([z,x,y,r],dim=1)
        out= self.FCz(input)
        for l in range(self.n_layers):
            preactivation=self.middle_layers[l](out)
            #warning, not tested with backprop
            new_out=_tnsr(torch.zeros(out.size()))
            for f,Fun in enumerate(self.Funs):
                mask=self.layer_masks[l*len(self.Funs)+f]
                new_out[:,mask] = Fun(preactivation[:,mask])
            #note: if you don't divide by 2 you get very pixelated results
            # due to quick saturation to 1 because of the sum
            out=(new_out+out)/2

        out=self.out_layer(out)
        return out.view(x.size(0),self.output_size)


    def mutate(self, copy_model=False):
        # mutate
        model=self
        if copy_model:
            model=copy.deepcopy(self)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                noise = _tnsr(normal(loc=0, scale=0.2, size=module.weight.data.size()))
                module.weight.data = module.weight.data + noise
                noise = _tnsr(normal(loc=0, scale=0.2, size=module.bias.data.size()))
                module.bias.data = module.bias.data+ noise

        # TODO also mutate the function mask
        #randomly flip a few bits from the mask
        # for layer in self.layer_masks.keys():
        #     self.layer_masks[layer] = \
        #         [torch.from_numpy(mask+ ).type(dtype) for mask in self.layer_masks[layer]]
        return model


    # #TODO
    # @staticmethod
    # def cross_over(CPPNs):
    #     model = copy.deepcopy(CPPNs[0])
    #     # create perlin mask
    #     tmp = OpenSimplex()
    #     masks={}
    #     for module in model.modules():
    #         if isinstance(module, nn.Linear):
    #             w_size=module.weight.size()
    #             b_size = module.bias.size()
    #
    #             x_range,y_range=np.linspace(0,1,w_size[0]),np.linspace(0,1,w_size[1])
    #
    #             # simplex noise should be generated over a floating range from 0 to 1
    #             perlin_weight=np.reshape([tmp.noise2d(x=x, y=y) for x,y in zip(np.meshgrid(x_range, y_range))],(w_size[0],w_size[1]))
    #             x_range, y_range = np.arange(0, b_size[0]), np.arange(0, b_size[1])
    #             perlin_bias = np.reshape([tmp.noise2d(x=x, y=y) for x, y in np.meshgrid(x_range, y_range)],
    #                                        (w_size[0], w_size[1]))
    #             #TODO
    #             #plot perlin stuff
    #
    #             # masks[module]=(perlin_weight,perlin_bias)
    #
    #
    #
    #     print(tmp.noise2d(x=10, y=10))
    #     # equalise historgram
    #     # asign interval of color randomly to parents
    #     return None
# from opensimplex import OpenSimplex
# import matplotlib.pyplot as plt
# import numpy as np
# tmp = OpenSimplex()
# w_size=[50,10]
# b_size=[10,10]
# x_range,y_range=np.linspace(0,1,w_size[0]),np.linspace(0,1,w_size[1])
# xv,yv=np.meshgrid(x_range, y_range)
#
#
# perlin_weight=np.reshape([tmp.noise2d(x=x, y=y) for x_row,y_row
#                           in zip(*np.meshgrid(x_range, y_range)) for x,y in zip(x_row,y_row)],
#                                            (w_size[0], w_size[1]))
#
# plt.imshow(perlin_weight)
# plt.show()
# x_range, y_range = np.arange(0, b_size[0]), np.arange(0, b_size[1])
# perlin_bias = np.reshape([tmp.noise2d(x=x, y=y) for x, y in np.meshgrid(x_range, y_range)],
#                                            (w_size[0], w_size[1]))
# plt.imshow(perlin_weight)


import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
class Node_CPPN(CPPN):


    def file_extension(self):
        return "ncppn"
    def save(self, file_path,fe="ncppn",dated=True):
        start_time = str(datetime.datetime.now()) if dated else ""
        fe="."+fe if fe is not "" else ""
        pickle_out = open(file_path+start_time+fe, "wb")
        save_graph = self.graph.copy()
        for node_name in save_graph.nodes:
            node=save_graph.nodes[node_name]
            node["value"] = None
            if "fun" in node:
                node["fun"] = node["fun"].name

        pickle.dump((self.z_scale,
                   self.z_size,
                   self.output_size,
                   self.nodes_in,
                   self.nodes_out,
                   [fun.name for fun in self.functions],
                   json_graph.node_link_data(save_graph),
                   [w.data.cpu().numpy() for w in self.all_weights]
                   ), pickle_out)
        pickle_out.close()
    @staticmethod
    def load(file_path,fe="ncppn"):
        fe = "." + fe if fe is not "" else ""
        pickle_in = open(file_path+fe, "rb")
        z_scale,z_size,output_size,nodes_in,nodes_out,fun_names,graph,all_weights = pickle.load(pickle_in)

        instance=Node_CPPN(z_size,fun_names,output_size,z_scale)

        instance.graph=json_graph.node_link_graph(graph)
        for node_name in instance.graph:
            node=instance.graph.nodes[node_name]
            if "fun" in node:
                node["fun"] = Fun(node["fun"])
        all_weights_pl= [nn.Parameter(torch.Tensor(w, device=device), requires_grad = False)
        for w in all_weights]
        instance.all_weights=nn.ParameterList(all_weights_pl)
        pickle_in.close()
        return instance
    def random_init_weights(self):
        self.reset()
        #connect fully
        self.add_edge()
        node=self.add_node()
        [self.init_edge(i, node) for i in ["x","y"]]
        [self.init_edge(node, i) for i in self.nodes_out]

    def reset(self):
        self.graph= nx.OrderedDiGraph()
        self.graph.add_nodes_from(self.nodes_in)
        self.graph.add_nodes_from(self.nodes_out, fun=Fun("sigm"))
    def __init__(self,z_size,fun_names,output_size=3,z_scale=1):
        super(Node_CPPN, self).__init__()
        self.z_scale=z_scale
        self.output_size=output_size
        self.z_size = z_size
        self.nodes_in = ['x', 'y', *['z' + str(i) for i in range(z_size)]]
        self.nodes_out = ['o' + str(i) for i in range(output_size)]
        self.functions = [Fun(f) for f in fun_names]

        # consistent ordering of node and edges, the nodes are in the order they are added
        self.graph = nx.OrderedDiGraph()
        self.graph.add_nodes_from(self.nodes_in)
        self.graph.add_nodes_from(self.nodes_out, fun=Fun("sigm"))
        self.all_weights=nn.ParameterList()
    def init_node(self,node, node_in, node_out):
        self.graph.add_node(node, fun=np.random.choice(self.functions))
        self.init_edge(node_in, node)
        self.init_edge(node, node_out)
    new_edge_weight_mu=1.7
    def _get_new_weight(self):
        weight = nn.Parameter(torch.Tensor(normal(0, self.new_edge_weight_mu, 1), device=device), requires_grad = False)
        self.all_weights.append(weight)
        return weight
    def init_edge(self,node_in, node_out):
        self.graph.add_edge(node_in, node_out, weight=self._get_new_weight())

    def add_node(self):
        node = str(len(list(self.graph.nodes)))
        # look for a random link
        edges = list(self.graph.edges)
        rnd_link = edges[np.random.randint(len(edges))]
        self.init_node(node, rnd_link[0], rnd_link[1])
        self.graph.add_node(node, fun=np.random.choice(self.functions))

        weight = self.graph.edges[rnd_link]["weight"]

        self.graph.add_edge(rnd_link[0], node, weight=self._get_new_weight())

        # the weight assigning scheme is one of many possibilities
        self.graph.add_edge(node, rnd_link[1], weight=weight)

        self.graph.remove_edge(rnd_link[0], rnd_link[1])
        return node


    mutation_var=0.5
    def mutate(self, copy_model=False):
        model = self
        if copy_model:
            # for node in self.graph.nodes:
            #     self.graph.nodes[node]['value'] = None
            model = copy.deepcopy(self)
        umt= np.random.random()
        if (umt>0.5):
            model.add_edge()
            model.add_node()
        else:
        #take random edge and change weight
            edge_idx=np.random.choice(range(len(model.graph.edges)))
            edge=list(model.graph.edges)[edge_idx]
            model.graph.edges[edge]["weight"]+=torch.Tensor(normal(0, self.mutation_var, 1))
        return model
    def draw_graph(self):
        # draw
        plt.subplot(111)
        nx.draw_circular(self.graph, with_labels=True, font_weight='bold')
        plt.show()
    def add_edge(self):
        # the output node should not have a path to the input node
        node_in = np.random.choice(list(set(self.graph.nodes) - set(self.nodes_out)))
        node_out = np.random.choice(list(set(self.graph.nodes) - set(self.nodes_in + [node_in])))

        while nx.has_path(self.graph, node_out, node_in):
            node_in = np.random.choice(list(set(self.graph.nodes) - set(self.nodes_out)), 1)[0]
            node_out = np.random.choice(list(set(self.graph.nodes) - set(self.nodes_in + [node_in])), 1)[0]
        self.init_edge(node_in, node_out)

    def prev(self, node):
        return [edge[0] for edge in self.graph.edges if edge[1] == node]

    def next(self, node):
        return [edge[1] for edge in self.graph.edges if edge[0] == node]
    def forward(self,X,Y,R,Z):
        self.graph.node['x']['value'] = X

        self.graph.node['y']['value'] = Y
        Z=Z/self.z_scale
        # only z can be partially unconnected
        nx.set_node_attributes(self.graph, dict([('z' + str(i), Z[:,i].unsqueeze(1)) for i in range(self.z_size)]), 'value')
        # find nodes that have all their inputs evaluated
        sorted_nodes = [node for node in nx.topological_sort(self.graph)
                        if node not in set(list(nx.isolates(self.graph)) + list(self.nodes_in))]
        #TODO this loop is slow and should be "baked" in some way (all consecutive functions)
        #use reduce maybe?
        for node in sorted_nodes:
            # the input value of the value vector a node gets from it's predecesor is its index as next node
            input = torch.sum(torch.stack([self.graph.nodes[n]['value'] * self.graph.edges[(n, node)]["weight"]
                                           for n in self.prev(node)], 1), 1)

            function = self.graph.nodes[node]['fun']
            # sum the weighted values of all previous nodes
            self.graph.nodes[node]['value'] = function(input)

        get_value = lambda c: self.graph.nodes[c]["value"] \
            if "value" in self.graph.nodes[c] else torch.zeros(X.size(0),device=device)
        return torch.stack([get_value(ni) for ni in self.nodes_out],1)

#crossover from original paper: nodes/connections present in both parents are randomly selected from either parent
# nodes/connections only present in 1 parent will always be added


