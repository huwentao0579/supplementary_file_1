import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict



def pad_and_mask(batch_x):
    max_len = max(len(x) for x in batch_x)
    batch_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in batch_x]
    batch_x = [list(x) + [0] * (max_len - len(x)) for x in batch_x]
    return batch_x, batch_mask


class DataGenerator:
    def __init__(self, args):
        print("Loading data...")

        self.tuples = sorted([
            eval(eachline) for eachline in open(args.data_filename.format("_train"), 'r').readlines()
        ], key=lambda k: len(k))
        self.total_tup_num = len(self.tuples)

        self.val_tuples = sorted([
            eval(eachline) for eachline in open(args.data_filename.format("_val"), 'r').readlines()
        ], key=lambda k: len(k))
        self.val_tup_num = len(self.val_tuples)

        self.real_tuples = self.tuples[:]
        self.real_val_tuples = self.val_tuples[:]

        self.args = args
        print("{} tuples loading complete.".format(self.total_tup_num))

        self.tup_sd = {idx: [0] for idx, tup in enumerate(self.tuples)}
        self.val_tup_sd = {idx: [0] for idx, tup in enumerate(self.val_tuples)}

        self.map_size = args.map_size
        self.sd_index = self.construct_sd_index()
        self.sd_ids = {sd: i for i, sd in enumerate(self.sd_index.keys())}
        print("Totally {} sd-pairs".format(len(self.sd_ids)))

        self.tup_sd_cluster = self.tup_sd
        self.val_tup_sd_cluster = self.val_tup_sd
        # self.outliers = [142,143,144,145,146,147]
        # self.outliers = [4970, 4971, 4972, 4973, 4974, 4975, 4976, 4977, 4978, 4979, 4980, 4981, 4982, 4983, 4984, 4985, 4986, 4987, 4988, 4989, 4990, 4991, 4992, 4993, 4994, 4995, 4996, 4997, 4998, 4999]
        self.outliers = [212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234]
    def make_model_dir(self):
        if hasattr(self.args, 'rnn_size'):
            model_dir = './models/{}_{}_{}/'.format(
                self.args.model_type, self.args.x_latent_size, self.args.rnn_size)
        elif hasattr(self.args, 'x_latent_size'):
            model_dir = './models/{}_{}/'.format(
                self.args.model_type, self.args.x_latent_size)
        else:
            model_dir = './models/{}/'.format(self.args.model_type)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        return model_dir

    def construct_sd_index(self):
        sd_index = defaultdict(list)
        for idx, tup in enumerate(self.tuples):
            sd_index[0].append(idx)
        return sd_index


    def next_batch(self, batch_size, partial_ratio=1.0, sd=False):
        anchor_idx = np.random.randint(0, self.total_tup_num)
        shortest_idx = max(0, anchor_idx - batch_size * 2)
        longest_idx = min(self.total_tup_num, anchor_idx + batch_size * 2)
        batch_idx = np.random.randint(shortest_idx, longest_idx, size=batch_size)
        batch_tuples = []
        batch_s, batch_d = [], []
        for tid in batch_idx:
            partial = int(len(self.tuples[tid]) * partial_ratio)
            batch_tuples.append(self.tuples[tid][:partial])
            batch_s.append(self.tup_sd_cluster[tid][0])
        batch_seq_length = [len(tup) for tup in batch_tuples]
        batch_x, batch_mask = pad_and_mask(batch_tuples)
        if "sd" in self.args.model_type or sd is True:
            return [batch_x, batch_mask, batch_seq_length], [batch_s, batch_d]
        else:
            return [batch_x, batch_mask, batch_seq_length]

    def iterate_all_data(self, batch_size, partial_ratio=1.0, purpose='train'):
        if purpose == 'train':
            tuples = self.tuples
            tup_num = self.total_tup_num
            tup_sd_cluster = self.tup_sd_cluster
        elif purpose == "val":
            tuples = self.val_tuples
            tup_num = self.val_tup_num
            tup_sd_cluster = self.val_tup_sd_cluster
        else:
            tuples = None
            tup_num = None
            tup_sd_cluster = None

        for batch_idx in range(0, tup_num, batch_size):
            batch_tuples = []
            batch_s, batch_d = [], []
            for tid in range(batch_idx, min(batch_idx + batch_size, tup_num)):
                partial = int(len(tuples[tid]) * partial_ratio)
                batch_tuples.append(tuples[tid][:partial])
                batch_s.append(tup_sd_cluster[tid][0])
            batch_seq_length = [len(tup) for tup in batch_tuples]
            batch_x, batch_mask = pad_and_mask(batch_tuples)
            if "sd" in self.args.model_type:
                yield [batch_x, batch_mask, batch_seq_length], [batch_s, batch_d]
            else:
                yield [batch_x, batch_mask, batch_seq_length, batch_x]

    def iterate_error_data(self, batch_size, error_data, idx, Min, max, partial_ratio=1.0):
        tuples = []
        for i in range(Min, max+1):
            error_data[idx] = i
            tp = error_data.copy()
            tuples.append(tp)
        tup_num = 1 + max - Min
        for batch_idx in range(0, tup_num, batch_size):
            batch_tuples = []
            for tid in range(batch_idx, min(batch_idx + batch_size, tup_num)):
                partial = int(len(tuples[tid]) * partial_ratio)
                batch_tuples.append(tuples[tid][:partial])
            batch_seq_length = [len(tup) for tup in batch_tuples]
            batch_x, batch_mask = pad_and_mask(batch_tuples)
            yield [batch_x, batch_mask, batch_seq_length, batch_x]

    def _perturb_point(self, point, level, offset=None):
        map_size = self.map_size
        x, y = int(point // map_size[1]), int(point % map_size[1])
        if offset is None:
            offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
            x_offset, y_offset = offset[np.random.randint(0, len(offset))]
        else:
            x_offset, y_offset = offset
        if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
            x += x_offset * level
            y += y_offset * level
        return int(x * map_size[1] + y)

    def perturb_batch(self, batch_x, level, prob):
        noisy_batch_x = []
        for tup in batch_x:
            noisy_batch_x.append([tup[0]] + [self._perturb_point(p, level)
                                 if not p == 0 and np.random.random() < prob else p
                                 for p in tup[1:-1]] + [tup[-1]])
        return noisy_batch_x

    def pan_batch(self, batch_x, level, prob, vary=False):
        map_size = self.map_size
        noisy_batch_x = []
        if vary:
            level += np.random.randint(-2, 3)
            if np.random.random() > 0.5:
                prob += 0.2 * np.random.random()
            else:
                prob -= 0.2 * np.random.random()
        for tup in batch_x:
            anomaly_len = int((len(tup) - 2) * prob)
            anomaly_st_loc = np.random.randint(1, len(tup) - anomaly_len - 1)
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            offset = [int(tup[anomaly_st_loc] // map_size[1]) - int(tup[anomaly_ed_loc] // map_size[1]),
                      int(tup[anomaly_st_loc] % map_size[1]) - int(tup[anomaly_ed_loc] % map_size[1])]
            if offset[0] == 0: div0 = 1
            else: div0 = abs(offset[0])
            if offset[1] == 0: div1 = 1
            else: div1 = abs(offset[1])

            if np.random.random() < 0.5:
                offset = [-offset[0] / div0, offset[1] / div1]
            else:
                offset = [offset[0] / div0, -offset[1] / div1]

            noisy_batch_x.append(tup[:anomaly_st_loc] +
                                 [self._perturb_point(p, level, offset) for p in tup[anomaly_st_loc:anomaly_ed_loc]] +
                                 tup[anomaly_ed_loc:])
        return noisy_batch_x

    def visualize(self, tup, c, alpha=1.0, lw=3, ls='-'):
        map_size = self.map_size
        gx, gy = [], []
        for p in tup:
            if not p == 0:
                gy.append(int(p // map_size[1]))
                gx.append(int(p % map_size[1]))
        plt.plot(gx, gy, color=c, linestyle=ls, lw=lw, alpha=alpha)

    def visualize_outlier_example(self, num, defult_idx=None):
        outlier_items = list(self.outliers)
        np.random.shuffle(outlier_items)
        if defult_idx is not None:
            selected_outliers = [(idx, self.outliers[idx]) for idx in defult_idx]
        else:
            selected_outliers = outlier_items[:num]
        for idx, o in selected_outliers:
            print(idx)
            self.visualize(o, c='r', alpha=1.0)
            self.visualize(self.real_tuples[idx], c='b', alpha=0.5)
        # plt.xlim(0, self.args.map_size[1])
        # plt.ylim(0, self.args.map_size[0])
        plt.show()

    def __spatial_augmentation(self):
        self.tuples = [self.perturb_batch([tup], level=1, prob=0.3)[0]
                             for tup in self.real_tuples]
        self.val_tuples = [self.perturb_batch([tup], level=1, prob=0.3)[0]
                                 for tup in self.real_val_tuples]

    def __sd_clustering(self):
        from sklearn.cluster import KMeans
        args = self.args

        tup_sd = np.array(list(self.tup_sd.values())).reshape(-1)
        sd_locs = np.array([[int(p // args.map_size[1]), int(p % args.map_size[1])] for p in tup_sd])
        val_tup_sd = np.array(list(self.val_tup_sd.values())).reshape(-1)
        val_sd_locs = np.array([[int(p // args.map_size[1]), int(p % args.map_size[1])] for p in val_tup_sd])

        self.kmeans = kmeans = KMeans(n_clusters=args.sd_cluster_num).fit(sd_locs)
        sd_clusters = kmeans.transform(sd_locs).reshape([-1, 2, args.sd_cluster_num]).tolist()
        self.tup_sd_cluster = dict(zip(self.tup_sd.keys(), sd_clusters))
        val_sd_clusters = kmeans.transform(val_sd_locs).reshape([-1, 2, args.sd_cluster_num]).tolist()
        self.val_tup_sd_cluster = dict(zip(self.val_tup_sd.keys(), val_sd_clusters))
        self.__save_clustering()

    def __save_clustering(self):
        model_dir = self.make_model_dir()
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        with open(model_dir + 'sd_clusters.pkl', 'wb') as fp:
            pickle.dump([self.kmeans,
                         self.tup_sd_cluster,
                         self.val_tup_sd_cluster], fp)

    def __load_clustering(self):
        model_dir = self.make_model_dir()
        with open(model_dir + 'sd_clusters.pkl', 'rb') as fp:
            self.kmeans, self.tup_sd_cluster, self.val_tup_sd_cluster = pickle.load(fp, encoding='latin1')