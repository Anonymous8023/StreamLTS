import warnings
from typing import Dict

import torch

from src.modules import BaseModule
from src.modules.utils.me_utils import update_me_essentials


class NaiveFusion(BaseModule):
    def __init__(self, data_info, stride, **kwargs):
        super(NaiveFusion, self).__init__(**kwargs)
        update_me_essentials(self, data_info, stride=stride)
        self.d = len(self.voxel_size)

    def forward(self, ego_feats, coop_feats=None, **kwargs):
        fused_feat = []
        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            coor = [ego_feat[f'p{self.stride}']['coor']]
            feat = [ego_feat[f'p{self.stride}']['feat']]
            ctr = [ego_feat[f'p{self.stride}']['ctr']]
            # fuse coop to ego
            for cpfeat in coop_feat.values():
                if 'pts_feat' not in cpfeat:
                    continue
                coor.append(cpfeat['pts_feat'][f'p{self.stride}']['coor'])
                feat.append(cpfeat['pts_feat'][f'p{self.stride}']['feat'])
                ctr.append(cpfeat['pts_feat'][f'p{self.stride}']['ctr'])
            coor = torch.cat(coor, dim=0)
            feat = torch.cat(feat, dim=0)
            ctr = torch.cat(ctr, dim=0)
            fused_feat.append({
                f'p{self.stride}': {
                    'coor': coor,
                    'feat': feat,
                    'ctr': ctr,
                }
            })
        return self.format_output(fused_feat)

    def format_output(self, output):
        return {self.scatter_keys[0]: output}





