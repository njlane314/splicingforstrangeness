from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import math
import pickle as pk
from numpy.random import uniform
from glob import glob


def compute_data_ranges(files):
    min_vals, max_vals = None, None
    for file in files:
        data = torch.load(file)
        hits = data['reco_hits']
        min_vals = np.min(hits, axis=0) if min_vals is None else np.minimum(min_vals, np.min(hits, axis=0))
        max_vals = np.max(hits, axis=0) if max_vals is None else np.maximum(max_vals, np.max(hits, axis=0))
    return min_vals[:3], max_vals[:3]


class HitAssociationTransformer(nn.Module):
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, num_particles, dim_feedforward=512, dropout=0.1):
        super(HitAssociationTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.proj_input = nn.Linear(input_size, d_model)
        self.classifier = nn.Linear(d_model, num_particles)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self, init_range=0.1):
        self.proj_input.bias.data.zero_()
        self.proj_input.weight.data.uniform_(-init_range, init_range)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor, src_mask: Tensor = None, src_padding_mask: Tensor = None):
        src_emb = self.proj_input(src)
        encoded_src = self.transformer_encoder(src=src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        particle_logits = self.classifier(self.dropout(encoded_src))
        return particle_logits


class TrajectoryFittingTransformer(nn.Module):
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward=512, dropout=0.1):
        super(TrajectoryFittingTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.proj_input = nn.Linear(input_size, d_model)
        self.decoder = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self, init_range=0.1):
        self.proj_input.bias.data.zero_()
        self.proj_input.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor, src_mask: Tensor = None, src_padding_mask: Tensor = None):
        src_emb = self.proj_input(src)
        encoded_src = self.transformer_encoder(src=src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        trajectory_points = self.decoder(self.dropout(encoded_src))
        return trajectory_points


class MomentumParticleFilter:
    def __init__(self, all_measurements, true_nodes, nb_particles, data_files):
        self.particles = None
        self.nb_meas = None
        self.weights = None
        self.dim = 5
        self.N = nb_particles

        self.data_files = data_files
        self.detector_min, self.detector_max = compute_data_ranges(self.data_files)
        self.cube_size = (self.detector_max - self.detector_min) / 100  # Example scaling for cube size

        with open("histogram_path.pkl", "rb") as fd:
            self.H, self.edges = pk.load(fd)

        self.forward_track = {"avg": [], "cov": []}
        self.backward_track = {"avg": [], "cov": []}

        self.stateAvg = np.zeros(shape=(self.dim,))
        self.stateCov = np.zeros(shape=(self.dim, self.dim))

        len_x = all_measurements[:, 0].max() - all_measurements[:, 0].min()
        len_y = all_measurements[:, 1].max() - all_measurements[:, 1].min()
        len_z = all_measurements[:, 2].max() - all_measurements[:, 2].min()
        self.axis = np.argmax([len_x, len_y, len_z])
        self.allMeasurements = all_measurements[all_measurements[:, self.axis].argsort()]
        self.true_nodes = true_nodes[true_nodes[:, self.axis].argsort()]
        self.dir = 1 if all_measurements[-1, self.axis] > all_measurements[0, self.axis] else -1

        self.failed = False

    def make_prior(self, forward=True):
        self.particles = np.zeros(shape=(self.N, self.dim))
        cans = (self.allMeasurements[:, self.axis] == self.allMeasurements[0, self.axis])
        self.nb_meas = cans.sum()
        best_cand = (self.allMeasurements[cans, 3]).argmax()
        best_cand = np.full(shape=(self.N,), fill_value=best_cand)
        self.particles[:, :3] = self.allMeasurements[best_cand, :3] + uniform(-self.cube_size, self.cube_size, size=(self.N, 3))
        self.particles[:, 3] = self.allMeasurements[best_cand, 3]
        self.particles[:, -1] = np.linspace(50, 1000, self.N)
        self.weights = np.ones(self.N) / self.N
        return

    def makeAverage(self):
        self.stateCov.fill(0)
        self.stateAvg[:] = np.average(self.particles, axis=0, weights=self.weights)
        diff = (self.particles - self.stateAvg).reshape(self.N, self.dim, 1)
        self.stateCov += (np.matmul(diff, diff.swapaxes(2, 1)).T * self.weights).sum(axis=-1)
        return

    def propagate(self, hits=15):
        max_nb_meas = min(self.nb_meas + hits, len(self.allMeasurements))
        cans = np.random.choice(range(self.nb_meas, max_nb_meas), self.N)
        self.particles[:, :3] = self.allMeasurements[cans, :3] + uniform(-self.cube_size, self.cube_size, size=(self.N, 3))
        self.particles[:, 3] = self.allMeasurements[cans, 3]
        return

    def likelihood(self, forward=True):
        delta_xyz = self.particles[:, :3] - self.stateAvg[:3]
        delta_x, delta_y, delta_z = delta_xyz[:, 0], delta_xyz[:, 1], delta_xyz[:, 2]
        _, delta_theta, _ = self.cart2spherical(delta_xyz)
        delta_pe = (np.log(self.particles[:, 3]) - np.log(self.stateAvg[3])).reshape(-1,)
        delta_momentum = self.particles[:, -1] - self.stateAvg[-1]
        if not forward:
            delta_x, delta_y, delta_z, delta_theta, delta_pe, delta_momentum = -delta_x, -delta_y, -delta_z, -delta_theta, -delta_pe, -delta_momentum
        indexes = [np.digitize(delta, edges) - 1 for delta, edges in zip([delta_x, delta_y, delta_z, delta_theta, delta_pe, delta_momentum], self.edges)]
        wrong_values = [np.logical_or(i == -1, i == h_dim) for i, h_dim in zip(indexes, self.H.shape)]
        for idx, wrong in zip(indexes, wrong_values):
            idx[wrong] = 0
        res = self.H[tuple(indexes)]
        res[np.logical_or.reduce(wrong_values)] = 0
        if res.sum() == 0:
            res[:] = np.random.normal(loc=0, scale=1, size=len(res))
            self.failed = True
        return res

    def update_particles(self, forward=True):
        self.propagate()
        self.weights = self.likelihood(forward)
        self.weights /= self.weights.sum()
        selected_xyz = self.particles[self.weights.argmax(), self.axis]
        if self.dir == 1:
            self.nb_meas = (self.allMeasurements[:, self.axis] < selected_xyz).sum()
        elif self.dir == -1:
            self.nb_meas = (self.allMeasurements[:, self.axis] > selected_xyz).sum()
        return

    def forward_backward_smoothing(self):
        cubes_for = ((self.forward_track["avg"][:, self.axis] - self.detector_min[self.axis]) / (self.cube_size[self.axis] * 2)).astype(int)
        cubes_bac = ((self.backward_track["avg"][:, self.axis] - self.detector_min[self.axis]) / (self.cube_size[self.axis] * 2)).astype(int)
        all_cubes = np.unique(np.concatenate((cubes_for, cubes_bac), axis=0))
        final_track = np.zeros(shape=(all_cubes.shape[0], 5))
        influence = 1.0 / 5.0
        for i, cube_id in enumerate(all_cubes):
            for_idx = np.where(cubes_for == cube_id)[0]
            bac_idx = np.where(cubes_bac == cube_id)[0]
            for_avg, bac_avg = self.forward_track["avg"][for_idx], self.backward_track["avg"][bac_idx]
            for_cov, bac_cov = self.forward_track["cov"][for_idx], self.backward_track["cov"][bac_idx]
            for_cov[for_cov == 0], bac_cov[bac_cov == 0] = 1e-7, 1e-7
            r, w_sum = 0, 0
            for j, fMeas in enumerate(for_idx):
                forw_err = np.linalg.inv(for_cov[j])
                forward_weight = influence * (fMeas + 1.0)
                w = forw_err.diagonal() * (1.0 - math.exp(-0.5 * forward_weight ** 2))
                r += w * for_avg[j]
                w_sum += w
            for j, bMeas in enumerate(bac_idx):
                back_err = np.linalg.inv(bac_cov[j])
                backward_weight = influence * (bMeas + 1.0)
                w = back_err.diagonal() * (1.0 - math.exp(-0.5 * backward_weight ** 2))
                r += w * bac_avg[j]
                w_sum += w
            final_track[i] = r / w_sum
        return final_track

    def estimate_initial_momentum(self):
        return self.particles[:, -1].mean()

    def run_filter(self):
        self.make_prior()
        self.makeAverage()
        self.forward_track["avg"].append(self.stateAvg.copy())
        self.forward_track["cov"].append(self.stateCov.copy())
        for _ in range(len(self.allMeasurements)):
            if self.nb_meas >= len(self.allMeasurements) - 1:
                break
            self.update_particles(forward=True)
            self.makeAverage()
            self.forward_track["avg"].append(self.stateAvg.copy())
            self.forward_track["cov"].append(self.stateCov.copy())
        self.forward_track["avg"] = np.array(self.forward_track["avg"])
        self.forward_track["cov"] = np.array(self.forward_track["cov"])
        self.allMeasurements = self.allMeasurements[::-1]
        self.dir *= -1
        self.make_prior()
        self.makeAverage()
        self.backward_track["avg"].append(self.stateAvg.copy())
        self.backward_track["cov"].append(self.stateCov.copy())
        for _ in range(len(self.allMeasurements)):
            if self.nb_meas >= len(self.allMeasurements) - 1:
                break
            self.update_particles(forward=False)
            self.makeAverage()
            self.backward_track["avg"].append(self.stateAvg.copy())
            self.backward_track["cov"].append(self.stateCov.copy())
        self.backward_track["avg"] = np.array(self.backward_track["avg"])
        self.backward_track["cov"] = np.array(self.backward_track["cov"])
        final_track = self.forward_backward_smoothing()
        initial_momentum = self.estimate_initial_momentum()
        return final_track, initial_momentum, self.failed
