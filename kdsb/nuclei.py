import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pdb import *
from pathlib import Path
from skimage.io import imread

#export
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.vision.all import *
# from PIL import Images

VAL_IMAGE_IDS = ['8f6e49e474ebb649a1e99662243d51a46cc9ba0c9c8f1efe2e2b662a81b48de1',
 '1740b0a67ca337ea31648b57c81bcfbb841c7bb5cad185199a9f4da596d531b9',
 '6d327ab4f0e3604fa6e9b8041c7e6db86ab809890d886c691f6e59c9168b7fbe',
 'a4c729efb5059893a8b62c7abeba171cb516836f8a20468f6b176dfe2f6f84d1',
 '5afb7932e9c7328f4fb1d7a8166a3699d6cdc5192b93758a75e9956f1513c5a3',
 'cbff60361ded0570e5d50429a1aa51d81471819bc9b38359f03cfef76de0038c',
 '6bc8cda54f5b66a2a27d962ac219f8075bf7cc43b87ba0c9e776404370429e80',
 '62570c4ff1c5ab6d9d383aba9f25e604768520b4266afd40fdf4734a694c8bc3',
 '0b2e702f90aee4fff2bc6e4326308d50cf04701082e718d4f831c8959fbcda93',
 '13f2bec0a24c70345372febb14c4352877b1b6c1b01896246048e83c345c0914',
 'b6edad733399c83c8eb7a59c0d37b54e10cc0d59894e39ff843884d84f61dee1',
 'bf566e75d5cb0196de4139573f8bbbda0fa38d5048edf7267fe8793dcc094a66',
 'e52960d31f8bddf85400259beb4521383f5ceface1080be3429f2f926cc9b5c2',
 'ddf1bf458312de2895dd9cc5ce7ec9d334ad54c35edc96ad6001d20b1d8588d8',
 'a7f6194ddbeaefb1da571226a97785d09ccafc5893ce3c77078d2040bccfcb77',
 'dbbfe08a52688d0ac8de9161cbb17cb201e3991aacab8ab8a77fe0e203a69481',
 '3b0709483b1e86449cc355bb797e841117ba178c6ae1ed955384f4da6486aa20',
 'aa4d989d262c618ac2793579e200cc71b3767f84698ae5f669867f23cdfe2568',
 'c0f172831b8017c769ff0e80f85b096ac939e79de3d524e0826fbb95221365da',
 'ebc18868864ad075548cc1784f4f9a237bb98335f9645ee727dac8332a3e3716',
 '66236902b874b7e4b3891db63a69f6d56f6edcec6aca7ba3c6871d73e7b4c34f',
 'e9b8ad127f2163438b6236c74938f43d7b4863aaf39a16367f4af59bfd96597b',
 'df9a4212ecb67bb4e58eba62f293b91f9d6f1dde73e38fa891c75661d419fc97',
 '2b50b1e3fa5c5aa39bc84ebfaea9961b7199c4d2488ae0b48d0b3459807d59d2',
 'b6d50fa22380ae3a7e8c52c5bc44a254e7b2596fd8927980dbe2c160cb5689b5',
 '6fb82031f7fc5f4fa6e0bc2ef3421db19036b5c2cdd2725009ab465d66d61d72',
 'd4d88391bc399a3715440d4da9f8b7a973e010dc1edd9551df2e5a538685add5',
 'eb1df8ed879d04b36980b0958a0e8fc446ad08c0bdcf3b5f42e3db023187c7e5',
 'dad607a203483439fcbc2acecd0a39fb5e5a94a32a94348f5c802c79cfeb6e7c',
 'a022908f1b7880838dbc0411e50828e64b4f5e0263afdf04295e30bb2ff58005',
 '9c95eae11da041189e84cda20bdfb75716a6594684de4b6ce12a9aaadbb874c9',
 'b3bfd873fca7ff9b2e90f507dfdbe165bb8c153399b6ba5829aa59bae677a91d',
 'dec1764c00e8b3c4bf1fc7a2fda341279218ff894186b0c2664128348683c757',
 'b76ff33ae9da28f9cd8bdce465d45f1eca399db3ffa83847535708e0d511fe38',
 '57bd029b19c1b382bef9db3ac14f13ea85e36a6053b92e46caedee95c05847ab',
 '1db1cddf28e305c9478519cfac144eee2242183fe59061f1f15487e925e8f5b5',
 '6bd18a218d25247dc456aed124c066a6397fb93086e860e4d04014bfa9c9555d',
 'd7d12a2acc47a94961aeb56fd56e8a0873016af75f5dd10915de9db8af8e4f5e',
 'da79a2b105f055ce75404a04bf53bbb51d518d9381af7b4ac714c137f101d920',
 '7798ca1ddb3133563e290c36228bc8f8f3c9f224e096f442ef0653856662d121',
 '8d29c5a03e0560c8f9338e8eb7bccf47930149c8173f9ba4b9279fb87d86cf6d',
 '33a5b0ff232b425796ee6a9dd5b516ff9aad54ca723b4ec490bf5cd9b2e2a731',
 'c3bec1066aae20f48b82975e7e8b684cd67635a8baf211e4d9e3e13bc54c5d06',
 '4829177d0b36abdd92c4ef0c7834cbc49f95232076bdd7e828f1f7cbb5ed80ec',
 'f26f4c2c70c38fe12e00d5a814d5116691f2ca548908126923fd76ddd665ed24',
 '4185b9369fc8bdcc7e7c68f2129b9a7442237cd0f836a4b6d13ef64bf0ef572a',
 '1815cf307859b3e13669041d181aa3b3dbbac1a95aef4c42164b223110c09168',
 'fc9269fb2e651cd4a32b65ae164f79b0a2ea823e0a83508c85d7985a6bed43cf',
 'a102535b0e88374bea4a1cfd9ee7cb3822ff54f4ab2a9845d428ec22f9ee2288',
 '8d05fb18ee0cda107d56735cafa6197a31884e0a5092dc6d41760fb92ae23ab4',
 '6f8197baf738986a1ec3b6ba92b567863d897a739376b7cec5599ad6cecafdfc',
 '0bf4b144167694b6846d584cf52c458f34f28fcae75328a2a096c8214e01c0d0',
 '3d0ca3498d97edebd28dbc7035eced40baa4af199af09cbb7251792accaa69fe',
 '8a65e41c630d85c0004ce1772ff66fbc87aca34cb165f695255b39343fcfc832',
 '1a75e9f15481d11084fe66bc2a5afac6dc5bec20ed56a7351a6d65ef0fe8762b',
 'a3a1b8f9794ef589b71faa9f35fd97ad6761c4488718fbcf766e95e31afa8606',
 '5488e8df5440ee5161fdfae3aeccd2ee396636430065c90e3f1f73870a975991',
 '8cdbdda8b3a64c97409c0160bcfb06eb8e876cedc3691aa63ca16dbafae6f948',
 'c2a646a819f59a4e816e0ee8ea00ba10d5de9ac20b5a435c41192637790dabee',
 '136000dc18fa6def2d6c98d4d0b2084d13c22eaffe82e26c665bcaa2a9e51261',
 'a9d884ba0929dac87c2052ce5b15034163685317d7cff45c40b0f7bd9bd4d9e7',
 '831218e6a1a54b23d4be56c5799854e7eb978811b89215319dc138900bd563e6',
 '2dd3356f2dcf470aec4003800744dfec6490e75d88011e1d835f4f3d60f88e7a',
 '2ab91a4408860ae8339689ed9f87aa9359de1bdd4ca5c2eab7fff7724dbd6707',
 '1e61ecf354cb93a62a9561db87a53985fb54e001444f98112ed0fc623fad793e',
 '2ad489c11ed8b77a9d8a2339ac64ffc38e79281c03a2507db4688fd3186c0fe5',
 'be771d6831e3f8f1af4696bc08a582f163735db5baf9906e4729acc6a05e1187']

# bounding boxes

def bounding_boxes(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=lambda x: len(x))[-1]
    return cv2.boundingRect(cnt) # x, y, w, h

def bbox2center(bbox):
    bbox = np.array(bbox)
    x = ((bbox[:, 0] + bbox[:, 2]//2))
    y = ((bbox[:, 1] + bbox[:, 3]//2))
    return np.stack((x,y),axis=-1)

def get_pt_annotations(DATA_PATH):
    pts_raw_path = DATA_PATH/'pt_annotations.pkl'

    if pts_raw_path.exists():
        pt_annotations = pts_raw_path.load()
    else:
        pt_annotations = []
        for tid in progress_bar(train_ids):
            path = TRAIN_PATH/tid
            img_path = Path(tid)/'images'/f'{tid}.png'

            bboxes = []
            for mask_file in (path/'masks').glob('*'):
                mask = imread(str(mask_file))
                bbox = bounding_boxes(mask)
                bboxes.append(bbox)

            pt_annotations.append((img_path, bbox2center(bboxes)))

        (DATA_PATH/'pt_annotations.pkl').save(pt_annotations)
    return pt_annotations



# Gaussian


@patch
def affine_coord(x: TensorMask, mat=None, coord_tfm=None, sz=None, mode='nearest',
                 pad_mode=PadMode.Reflection, align_corners=True):
    add_dim = (x.ndim==3)
    if add_dim: x = x[:,None]
    res = TensorImage.affine_coord(x.float(), mat, coord_tfm, sz, mode, pad_mode, align_corners)#.long() - We use gaussian kernels. Mask must be float
    if add_dim: res = res[:,0]
    return TensorMask(res)


@IntToFloatTensor
def encodes(self, o:TensorMask ): return o


# taken from https://github.com/xingyizhou/CenterNet/blob/819e0d0dde02f7b8cb0644987a8d3a370aa8206a/src/lib/utils/image.py

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap



# Calculations

import cv2
from functools import partial

def to_pts(box): # x,y,w,h -> x1,y1,x2,y2
    x,y,w,h = box
    return x,y,x+w,y+h

def score(box, pred): # get prediction score
    x1,y1,x2,y2 = box
    return pred[:, y1:y2,x1:x2].max()

# https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def compute_ap(precision, recall):
    "Compute the average precision for `precision` and `recall` curve."
    recall = np.concatenate(([0.], list(recall), [1.]))
    precision = np.concatenate(([0.], list(precision), [0.]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    idx = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return ap

def euclidean_dist(boxA, boxB):
    def midpt(box):
        x1,y1,x2,y2 = box
        return (x1+x2)/2, (y1+y2)/2
    (xA,yA), (xB,yB) = midpt(boxA), midpt(boxB)
    return ((xB-xA)**2 + (yB-yA)**2)**0.5

def calc_pr(lbls, preds, pmaxs, min_sz=1, max_dist=5, iou_thresh=0.1, score_thresh=0.3):
    tps = []
    fps = []
    scores = []
    n_gts = []
    for lbl,pred,pmax in zip(lbls,preds,pmaxs):
        contours,hierarchy = cv2.findContours(pmax.max(0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pboxes = [cv2.boundingRect(cnt) for cnt in contours]
        pboxes = [to_pts(pbox) for pbox in pboxes if pbox[2] >= min_sz and pbox[3] >= min_sz] # only if width and height are greater than min_sz
        pboxes = [pbox for pbox in pboxes if score(pbox, pred) >= score_thresh]

        contours,hierarchy = cv2.findContours((lbl>=0.9).max(0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        lboxes = [to_pts(cv2.boundingRect(cnt)) for cnt in contours]
        
#         detected = []
#         ious = []
#         for pb in pboxes:
#             calc = [(bb_iou(lb, pb), lb) for lb in lboxes if lb not in detected]
#             if len(calc) == 0: ious.append(0)
#             else:
#                 iou, lb = max(calc)
#                 detected.append(lb)
#                 ious.append(iou)
#         tp = (np.array(ious) >= iou_thresh)
#         fp = ~tp
#         s = np.array([score(pb, pred) for pb in pboxes])
                
        detected = []
        dists = []
        for pb in pboxes:
            calc = [(euclidean_dist(lb, pb), lb) for lb in lboxes if lb not in detected]
            if len(calc) == 0:
                dists.append(1e10)
            else:
                dist, lb = min(calc)
                detected.append(lb)
                dists.append(dist)
        tp = (np.array(dists) < max_dist)
        fp = ~tp
        s = np.array([score(pb, pred) for pb in pboxes])
            
        
        n_gts.append(len(lboxes))
        tps.extend(tp.astype(np.uint8).tolist())
        fps.extend(fp.astype(np.uint8).tolist())
        scores.extend(s.tolist())
        
    res = sorted(zip(scores, tps, fps), key=lambda x: x[0], reverse=True)
    res = np.array(res)
    if len(res) == 0: res = np.zeros((1, 3))
    tp = res[:,1].cumsum(0)
    fp = res[:,2].cumsum(0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / sum(n_gts)
    return precision, recall




# misc
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# https://www.kaggle.com/hocop1/centernet-baseline

class FocalLoss2(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out, target):
        # Binary mask loss
        pred_mask = torch.sigmoid(out)
        target = target[:, None]
    #     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
        mask_loss = target * torch.log(pred_mask + 1e-12) + (1 - target) * torch.log(1 - pred_mask + 1e-12)
        mask_loss = -mask_loss.mean(0).sum()
        return mask_loss
