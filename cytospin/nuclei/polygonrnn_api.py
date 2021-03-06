# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/labelbox/3_PolygonRNN_API.ipynb (unless otherwise specified).

__all__ = ['polyrnn_path', 'device', 'get_args', 'get_data_loaders', 'Tool', 'args', 'tool', 'get_points']

# Cell
from .imports import *
from .labelbox_utils import *

# Cell
import sys
polyrnn_path = Path("../../polyrnn/code").absolute()
sys.path.append(str(polyrnn_path))

# Cell
import os
import numpy as np
import wget
import argparse
import base64
import json
import time
import torch

from Utils import utils
from DataProvider import cityscapes
from Models.Poly import polyrnnpp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Cell
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', required=True)
    parser.add_argument('--reload', required=True)
    parser.add_argument('--image_dir', default='static/images/')
    parser.add_argument('--port', type=int, default=5001)

    args = parser.parse_args(['--exp', str(polyrnn_path/'Experiments/tool.json'),
                              '--reload', str(polyrnn_path/'../models/ggnn_epoch5_step14000.pth'),
                              '--image_dir', str(polyrnn_path/'Tool/frontend/static/img/')])

    return args

def get_data_loaders(opts, DataProvider):
    print('Building dataloaders')
    data_loader = DataProvider(split='val', opts=opts['train_val'], mode='tool')

    return data_loader

# Cell
class Tool(object):
    def __init__(self, args):
        self.opts = json.load(open(args.exp, 'r'))
        self.image_dir = args.image_dir
        self.data_loader = get_data_loaders(self.opts['dataset'], cityscapes.DataProvider)
        self.model = polyrnnpp.PolyRNNpp(self.opts).to(device)
        self.model.reload(args.reload, strict=True)

    def get_grid_size(self, run_ggnn=True):
        if self.opts['use_ggnn'] and run_ggnn:
            grid_size = self.model.ggnn.ggnn_grid_size
        else:
            grid_size = self.model.encoder.feat_size

        return grid_size

    def annotation(self, instance, run_ggnn=False):
        with torch.no_grad():
            img = np.expand_dims(instance['img'], 0)
            img = torch.from_numpy(img).to(device)
            # Add batch dimension and make torch Tensor

            output = self.model(
                img,
                poly=None,
                fp_beam_size=5,
                lstm_beam_size=1,
                run_ggnn=run_ggnn
            )
            polys = output['pred_polys'].cpu().numpy()

        del(output)
        grid_size = self.get_grid_size(run_ggnn)
        return self.process_output(polys, instance,
            grid_size)

#     def fixing(self, instance, run_ggnn=False):
#             disabling feedback mode - since we're using delayed updates with labelbox

    def process_output(self, polys, instance, grid_size):
        poly = polys[0]
        poly = utils.get_masked_poly(poly, grid_size)
        poly = utils.class_to_xy(poly, grid_size)
        poly = utils.poly0g_to_poly01(poly, grid_size)
        poly = poly * instance['patch_w']
        poly = poly + instance['starting_point']

        torch.cuda.empty_cache()
        return [poly.astype(np.int).tolist()]

# Cell
args = get_args()
tool = Tool(args)
tool.model.eval()

# Cell
def get_points(bbox, img_path):
    component = {'poly': np.array([[-1., -1.]])}
    instance = {'image_id': 0,
                'img_path': img_path,
                'bbox': bbox2sz(bbox)}
    # component['poly'] = np.array([[-1., -1.]])
    instance = tool.data_loader.prepare_component(instance, component)
    return tool.annotation(instance), instance