import torch
from torch.utils.data import Dataset
import numpy as np
import random
import uproot
from glob import glob

class HitAssociationDataset(Dataset):
    def __init__(self, root, u_treename, v_treename, w_treename, seq_length=100, to_tensor=True, normalize=True, shuffle=False):
        self.root = root
        self.data_files = sorted(glob(f'{self.root}/*.root'))
        if shuffle:
            random.shuffle(self.data_files)
        self.u_treename = u_treename
        self.v_treename = v_treename
        self.w_treename = w_treename
        self.seq_length = seq_length
        self.to_tensor = to_tensor
        self.normalize = normalize

        if self.normalize:
            self.norm_params = self.compute_normalization_params()

    def __len__(self):
        return len(self.data_files)

    def compute_normalization_params(self):
        min_vals, max_vals = None, None
        for file in self.data_files:
            u_data, v_data, w_data = self.load_planes(file)
            hits_combined = np.vstack((u_data['x'], v_data['x'], w_data['x'], 
                                       u_data['z'], v_data['z'], w_data['z'],
                                       u_data['q'], v_data['q'], w_data['q']))
            min_vals = np.min(hits_combined, axis=0) if min_vals is None else np.minimum(min_vals, np.min(hits_combined, axis=0))
            max_vals = np.max(hits_combined, axis=0) if max_vals is None else np.maximum(max_vals, np.max(hits_combined, axis=0))
        return min_vals, max_vals

    def apply_norm(self, X):
        min_vals, max_vals = self.norm_params
        X[:, :3] = (X[:, :3] - min_vals[:3]) / (max_vals[:3] - min_vals[:3])
        if X.shape[1] > 3:
            X[:, 3] = (X[:, 3] - min_vals[3]) / (max_vals[3] - min_vals[3])

    def load_planes(self, filename):
        file = uproot.open(filename)
        u_data = self.load_plane(file, self.u_treename)
        v_data = self.load_plane(file, self.v_treename)
        w_data = self.load_plane(file, self.w_treename)
        file.close()
        return u_data, v_data, w_data

    def load_plane(self, file, treename):
        tree = file[treename]
        x = tree['x'].array(library="np")
        z = tree['z'].array(library="np")
        q = tree['q'].array(library="np")

        for i in range(len(x)):
            x[i] -= x[i][0]
            z[i] -= z[i][0]

        x_pad = self.pad_sequences(x)
        z_pad = self.pad_sequences(z)
        q_pad = self.pad_sequences(q)
        
        return {"x": x_pad, "z": z_pad, "q": q_pad, "is_good": tree["is_good"].array(library="np").astype(int)}

    def pad_sequences(self, values):
        return np.array([
            np.concatenate((value[:self.seq_length], np.zeros(max(0, self.seq_length - len(value)))))
            for value in values
        ])

    def __getitem__(self, idx):
        u_data, v_data, w_data = self.load_planes(self.data_files[idx])

        sequence = np.dstack((
            np.array([u_data['x'][idx], v_data['x'][idx], w_data['x'][idx]]),
            np.array([u_data['z'][idx], v_data['z'][idx], w_data['z'][idx]]),
            np.array([u_data['q'][idx], v_data['q'][idx], w_data['q'][idx]])
        ))

        if self.normalize:
            self.apply_norm(sequence)
        
        sequence = torch.tensor(sequence, dtype=torch.float32) if self.to_tensor else sequence
        label = torch.tensor(u_data['is_good'][idx], dtype=torch.long)  
        return sequence, label



class TrajectoryFittingDataset(Dataset):
    def __init__(self, hit_association_dataset, single_association_criteria):
        self.hit_association_dataset = hit_association_dataset
        self.single_association_criteria = single_association_criteria

    def __len__(self):
        return len(self.hit_association_dataset)

    def __getitem__(self, idx):
        sequence, label = self.hit_association_dataset[idx]
        single_association_hits = sequence[self.single_association_criteria(sequence)]
        
        return single_association_hits, label