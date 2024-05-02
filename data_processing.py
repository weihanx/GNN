import torch
import glob

import numpy as np
import networkx as nx
import os
from pathlib import Path

from pyScoreParser.musicxml_parser.mxp import MusicXMLDocument
from pyScoreParser.musicxml_parser.mxp.note import Note
import pyScoreParser.score_as_graph as score_graph
import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, HeteroData
from data_maps import *
from config import INTERVAL_EDGES


class HeterGraph(Dataset):
    def __init__(self, root, train_names=None, transform=None, pre_transform=None):
        """
        root: where my dataset should be stored: it will automatically saved at root/processed
        """
        self.train_names = train_names
        super(HeterGraph, self).__init__(root, transform, pre_transform)
        self.data_list = []

        for file_name in self.processed_file_names:
            file_path = os.path.join(self.processed_dir, file_name)
            if os.path.isfile(file_path):
                self.data_list.append(torch.load(file_path))
            else:
                print(f"Missing processed file: {file_path}")
        self.root = root

    def len(self):
        return len(self.processed_file_names)

    def __len__(self):
        return self.len()

    def get(self, idx):
        return self.data_list[idx]

    def __getitem__(self, idx):
        return self.get(idx)

    @property
    def processed_file_names(self):
        return [f'{i}_processed.pt' for i in range(1286)]

    def one_hot_convert(self, mapped_pitch, num_class):
        # number of samples, number of class
        one_hot_encoded = np.zeros((len(mapped_pitch), num_class))
        for i, pitch in enumerate(mapped_pitch):
            one_hot_encoded[i, pitch] = 1
        return self.to_float_tensor(one_hot_encoded)

    @staticmethod
    def to_float_tensor(array):
        if array.dtype == np.object_:
            array = np.array(array, dtype=float)
        return torch.tensor(array, dtype=torch.float)

    @staticmethod
    def add_interval_edges(notes: list[Note], edge_indices, intervals: list[int] = INTERVAL_EDGES):
        interval_up_from_to = {interval: [] for interval in intervals}
        interval_down_from_to = {interval: [] for interval in intervals}

        for i, current_note in enumerate(notes):
            ups_found = {interval: False for interval in intervals}
            downs_found = {interval: False for interval in intervals}
            for interval in intervals:
                for j, potential_next_note in enumerate(notes):
                    if potential_next_note.state_fixed.time_position <= current_note.state_fixed.time_position: continue
                    lower, upper = DIATONIC_TO_CHROMATIC_INTERVAL[interval]
                    if lower <= potential_next_note.pitch[1] - current_note.pitch[1] <= upper and not ups_found[interval]:
                        ups_found[interval] = True
                        interval_up_from_to[interval].append([i, j])
                    if -upper <= potential_next_note.pitch[1] - current_note.pitch[1] <= -lower and not downs_found[interval]:
                        downs_found[interval] = True
                        interval_down_from_to[interval].append([i, j])
                    if ups_found[interval] and downs_found[interval]:
                        break

        for interval in intervals:
            edge_indices[f"up{interval}"] = interval_up_from_to[interval]
            edge_indices[f"down{interval}"] = interval_down_from_to[interval]
        return edge_indices

    @staticmethod
    def extract_clusters(pkl_file, notes):
        with open(pkl_file, 'rb') as f:
            clusters = pickle.load(f)
        cluster_tuple = ()
        assert clusters[0].shape[0] == len(notes)

        for cluster in clusters:
            cur_cluster = torch.tensor(cluster, dtype=torch.float32)
            cluster_tuple = cluster_tuple + (cur_cluster,)
        return cluster_tuple

    def process_file_nodes(self, hetero_data, pyscoreparser_notes):
        offsets = [note.state_fixed.time_position for note in pyscoreparser_notes]
        durations = [note.note_duration.seconds for note in pyscoreparser_notes]

        node_features = {
            "midi": np.array([(note.pitch[1] - 21) / 88 for note in pyscoreparser_notes]),
            "pitch_class": [PITCH_CLASS_MAP[note.pitch[0]] for note in pyscoreparser_notes],
            "duration": np.array([duration / np.max(durations) for duration in durations]),
            "offsets": np.array([offset / np.max(offsets) for offset in offsets]),
        }
        node_features["midi"] = self.to_float_tensor(node_features["midi"]).unsqueeze(1)
        node_features["pitch_class"] = self.one_hot_convert(node_features["pitch_class"], len(PITCH_CLASS_MAP))
        node_features["duration"] = self.to_float_tensor(node_features["duration"]).unsqueeze(1)
        node_features["offsets"] = self.to_float_tensor(node_features["offsets"]).unsqueeze(1)

        note_features = torch.cat([feature for feature in node_features.values()], dim=1)

        notes_graph = score_graph.make_edge(pyscoreparser_notes)
        hetero_data['note'].x = note_features

        return hetero_data, notes_graph

    def process_file_edges(self, hetero_data, notes_graph, pyscoreparser_notes):
        edge_indices = {k: [] for k in [
            "onset",
            "voice",
            "forward",
            # "slur",
            # "sustain",
            "rest",
        ]}
        edge_indices = self.add_interval_edges(pyscoreparser_notes, edge_indices)

        for edge in notes_graph:
            from_to = edge[:2]
            edge_type = edge[2]
            if edge_type in edge_indices.keys():
                edge_indices[edge_type].append(from_to)
        edge_indices = {
            edge_type: torch.tensor(np.array(edges), dtype=torch.long).t().contiguous()
            for edge_type, edges in edge_indices.items()
        }
        edge_indices = [
            (('note', edge_type, 'note'), edge_indices)
            for edge_type, edge_indices in edge_indices.items()
        ]

        # Initialize edge weights to 1
        for edge_type, edge_index in edge_indices:
            if edge_index.numel() == 0:
                continue
            hetero_data[edge_type].edge_index = edge_index
            num_edges = hetero_data[edge_type].edge_index.shape[1]
            edge_weights = torch.ones(num_edges)
            hetero_data[edge_type].edge_attr = edge_weights

        return hetero_data

    def process_file(self, xml_file, pkl_file, index):
        xml_file = Path(xml_file)
        XMLDocument = MusicXMLDocument(str(xml_file))
        pyscoreparser_notes = XMLDocument.get_notes()

        hetero_data = HeteroData()
        hetero_data, notes_graph = self.process_file_nodes(hetero_data, pyscoreparser_notes)
        hetero_data = self.process_file_edges(hetero_data, notes_graph, pyscoreparser_notes)

        ground_truth_clusters = self.extract_clusters(pkl_file, pyscoreparser_notes)

        data_dict = {
            "name": str(xml_file).removesuffix('.xml'),
            "data": hetero_data,
            "cluster": ground_truth_clusters
        }
        torch.save(data_dict, os.path.join(self.processed_dir, f'{index}_processed.pt'))
        self.data_list.append(data_dict)

    def process(self):
        self.data_list = []
        index = 0
        for directory in self.train_names:
            pkl_files = []
            pkl_files.extend(glob.glob(f"{directory}/**/*.pkl", recursive=True))
            pkl_file = pkl_files[0]
            xml_files = []
            xml_files.extend(glob.glob(f"{directory}/**/*.xml", recursive=True))
            for xml_file in xml_files:
                if index % 100 == 0:
                    print(f"Processing file {xml_file}")
                self.process_file(xml_file, pkl_file, index)
                index += 1

    def hetero_to_networkx(self, obj_idx):
        edge_type_color = {
            'forward': 'red',
            'onset': 'green',
            'sustain': 'blue',
            'rest': 'yellow',
        }

        file_name = self.processed_file_names[obj_idx]
        file_path = os.path.join(self.processed_dir, file_name)
        hetero_data = torch.load(file_path)
        G = nx.DiGraph()

        # Add nodes
        for node_type in hetero_data.node_types:
            for node_id in range(hetero_data[node_type].NUM_NODES):
                G.add_node(f"{node_type}_{node_id}", type=node_type)

        # Add edges with colors
        for edge_type in hetero_data.edge_types:
            color = edge_type_color.get(edge_type[1], 'black')
            for source, target in hetero_data[edge_type].edge_index.t().numpy():
                src_node = f"{edge_type[0]}_{source}"
                tgt_node = f"{edge_type[2]}_{target}"
                G.add_edge(src_node, tgt_node, type=edge_type[1], color=color)

        return G


if __name__ == "__main__":
    with open("sck_samples.txt", "r") as file:
        train_names = file.readlines()
        train_names = [line.strip() for line in train_names]

    dataset = HeterGraph(root="processed/heterdatacleaned/", train_names=train_names)
    train_loader = DataLoader(dataset, batch_size=1)

    for data, sck_cluster_tuple in train_loader:
        print(f"data = {data}, {sck_cluster_tuple[0][0].shape}, {sck_cluster_tuple[1][0].shape}")
