import random
import numpy as np
from src.dataset.base_dataset import BaseDataset


class TacoodDataset(BaseDataset):
    """Sequential data loader."""
    def __init__(self, cfgs, mode):
        super().__init__(cfgs, mode)
        self.seq_len = cfgs['seq_len']
        self.rand_len = cfgs.get('rand_len', 0)
        self.seq_mode = cfgs.get('seq_mode', False)
        self.clean_seq = cfgs.get('clean_seq', False)

    def __getitem__(self, index):
        queue = []
        index_list = list(range(index - self.seq_len - self.rand_len + 1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.rand_len:])
        index_list.append(index)
        prev_scene_token = None
        prev_agents = None
        prev_i = None
        num_cav = None

        for i in index_list:
            i = max(0, i)
            input_dict = self.load_frame_data(i, prev_agents, prev_i)
            prev_i = i

            if not self.seq_mode:  # for sliding window only
                prev_exists = []
                prev_agents = []
                for tk in input_dict['scene_tokens']:
                    prev_agents.append(tk.split('.')[-1])
                    if prev_scene_token is not None and tk in prev_scene_token:
                        prev_exists.append(np.array([True]))
                    else:
                        prev_exists.append(np.array([False]))
                input_dict.update(dict(prev_exists=prev_exists))
                prev_scene_token = input_dict['scene_tokens']

            queue.append(input_dict)

        # remove frames not belong to the current sequence
        # and ensure all frames have the same ego id
        valid_idx_start = 0
        if self.clean_seq:
            ego_id = queue[-1]['valid_agent_ids'][0]
            for i in range(len(queue)):
                if queue[i]['valid_agent_ids'][0] != ego_id:
                    valid_idx_start = i + 1
        queue = {k: [q[k] for q in queue[valid_idx_start:]] for k in queue[0].keys()}
        return queue


