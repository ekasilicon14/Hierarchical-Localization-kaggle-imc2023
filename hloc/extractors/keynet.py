import sys
from pathlib import Path
from functools import partial
import torch
import torch.nn.functional as F
import kornia.feature as KF
import kornia as K

from ..utils.base_model import BaseModel

class KeyNetAffNetHardNetLayers(KF.LocalFeature):
    def __init__(
        self,
        num_features: int = 5000,
        upright: bool = False,
        device = torch.device('cpu'),
        scale_laf: float = 1.0,
    ):
        ori_module = KF.PassLAF() if upright else KF.LAFOrienter(angle_detector=KF.OriNet(False)).eval()
        if not upright:
            weights = torch.load('/kaggle/input/kornia-local-feature-weights/OriNet.pth')['state_dict']
            ori_module.angle_detector.load_state_dict(weights)
        detector = KF.KeyNetDetector(
            False, num_features=num_features, ori_module=ori_module, aff_module=KF.LAFAffNetShapeEstimator(False).eval()
        ).to(device)
        kn_weights = torch.load('/kaggle/input/kornia-local-feature-weights/keynet_pytorch.pth')['state_dict']
        detector.model.load_state_dict(kn_weights)
        affnet_weights = torch.load('/kaggle/input/kornia-local-feature-weights/AffNet.pth')['state_dict']
        detector.aff.load_state_dict(affnet_weights)
        
        hardnet = KF.HardNet(False).eval()
        hn_weights = torch.load('/kaggle/input/kornia-local-feature-weights/HardNetLib.pth')['state_dict']
        hardnet.load_state_dict(hn_weights)
        descriptor = KF.LAFDescriptor(hardnet, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor, scale_laf)

class KeyNetAffNetHardNet(BaseModel):
    default_conf = {
        'max_keypoints': 10000,
        'upright': False,
        'scale_laf': 1.0,
        'resize_small_edge_to': 600
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.feature = KeyNetAffNetHardNetLayers(conf['max_keypoints'], conf['upright'], 'cuda').to('cuda').eval()
        self.resize_small_edge_to = conf['resize_small_edge_to']

    def _forward(self, data):
        timage = data['image']

        orig_h, orig_w = timage.shape[-2:]

        if self.resize_small_edge_to is None:
            timg_resized = timage
        else:
            timg_resized = K.geometry.resize(timage, self.resize_small_edge_to, antialias=True)
        new_h, new_w = timg_resized.shape[2:]
        lafs, resps, descs = self.feature(K.color.rgb_to_grayscale(timg_resized))

        lafs[:,:,0,:] *= float(orig_w) / float(new_w)
        lafs[:,:,1,:] *= float(orig_h) / float(new_h)
        desc_dim = descs.shape[-1]
        kpts = torch.from_numpy(KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy())
        descs = torch.from_numpy(descs.reshape(-1, desc_dim).detach().cpu().numpy())
        resps = torch.flatten(resps)
        scales = torch.flatten(KF.get_laf_scale(lafs))
        oris = torch.flatten(KF.get_laf_orientation(lafs))

        return {
            'keypoints': kpts[None],
            'scales': scales[None],
            'oris': oris[None],
            'scores': resps[None],
            'descriptors': descs.T[None],
        }