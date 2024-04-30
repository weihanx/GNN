import torch
import glob
import warnings

from torch_geometric.utils import add_self_loops
import numpy as np
import networkx as nx
import os
from pathlib import Path
from pyScoreParser.musicxml_parser.mxp import MusicXMLDocument
import pyScoreParser.score_as_graph as score_graph
import pickle
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, HeteroData
from music21 import *
import xml.etree.ElementTree as ET


pitch_class_map = {
    'Cbb': 0,
    'Cb': 1,
    'C': 2,
    'C#': 3,
    'Cx': 4,
    'Dbb': 5,
    'Db': 6,
    'D': 7,
    'D#': 8,
    'Dx': 9,
    'Ebb': 10,
    'Eb': 11,
    'E': 12,
    'E#': 13,
    'Ex': 14,
    'Fbb': 15,
    'Fb': 16,
    'F': 17,
    'F#': 18,
    'Fx': 19,
    'Gbb': 20,
    'Gb': 21,
    'G': 22,
    'G#': 23,
    'Gx': 24,
    'Abb': 25,
    'Ab': 26,
    'A': 27,
    'A#': 28,
    'Ax': 29,
    'Bbb': 30,
    'Bb': 31,
    'B': 32,
    'B#': 33,
    'Bx': 34
}

midi_map = {
    '63': 0, '61': 1, '60': 2, '65': 3, '58': 4, '56': 5, '62': 6, '59': 7, '64': 8, '57': 9,
    '55': 10, '66': 11, '54': 12, '53': 13, '51': 14, '49': 15, '67': 16, '68': 17, '52': 18,
    '50': 19, '48': 20, '47': 21, '75': 22, '72': 23, '73': 24, '77': 25, '74': 26, '70': 27,
    '79': 28, '69': 29, '71': 30, '76': 31, '78': 32, '80': 33, '90': 34, '89': 35, '92': 36,
    '91': 37, '83': 38, '85': 39, '84': 40, '87': 41, '86': 42, '88': 43, '81': 44, '82': 45,
    '45': 46, '44': 47, '43': 48, '46': 49, '41': 50, '40': 51, '38': 52, '36': 53, '42': 54,
    '39': 55, '37': 56, '35': 57
}

duration_map = {
    '10080': 0, '5040': 1, '15120': 2, '2520': 3, '20160': 4, '7560': 5, '40320': 6,
    '30240': 7, '1260': 8, '1680': 9, '3360': 10, '13440': 11, '2': 12, '6': 13,
    '1': 14, '8': 15, '4': 16
}


class HeterGraph(Dataset):
    def __init__(self, root, train_names=None, transform=None, pre_transform=None, add_self_loops=False):
        """
        root: where my dataset should be stored: it will automatically saved at root/processed
        """
        self.train_names = train_names  #
        self.add_self_loops = add_self_loops
        # print(f"self.processed_dir = {self.processed_dir}") # homodata/processed
        super(HeterGraph, self).__init__(root, transform, pre_transform)  # invoke process when necessary
        # self.data_list = []
        self.data_list = []  # according to my class, each object is tuple(data, matrix)

        for file_name in self.processed_file_names:
            file_path = os.path.join(self.processed_dir, file_name)
            if os.path.isfile(file_path):
                self.data_list.append(torch.load(file_path))
            else:
                print(f"Missing processed file: {file_path}")
        self.root = root

    def len(self):
        # print(f"call len = {len(self.data_list)}")

        return len(self.processed_file_names)

    def get(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        data_tuple = self.get(idx)
        return data_tuple

    @property
    def processed_file_names(self):
        """
        If these files are found, processed will be skipped, make sure number of names match, I can shuffle
        """

        return [f'{i}_processed.pt' for i in range(1114)]

    def one_hot_convert(self, mapped_pitch, num_class):
        # number of samples, number of class
        one_hot_encoded = np.zeros((len(mapped_pitch), num_class))
        for i, pitch in enumerate(mapped_pitch):
            one_hot_encoded[i, pitch] = 1
        return one_hot_encoded

    def pad_cluster(self, final_size, cluster_tensor):
        current_height, _ = cluster_tensor.shape
        padding_top = 0  # No padding at the top
        padding_left = 0  # No padding on the left
        padding_bottom = final_size - current_height - padding_top  # pad new nodes, all zero
        padding_right = 0

        # Pad the tensor
        padded_cluster = F.pad(cluster_tensor, (padding_left, padding_right, padding_top, padding_bottom), 'constant',
                               0)

        return padded_cluster

    def normalize(self, array):
        return (array - array.min()) / (array.max() - array.min())

    def to_float_tensor(self, array):
        # Convert array to a numeric dtype if it's not already, then to a torch.tensor
        if array.dtype == np.object_:
            array = np.array(array, dtype=float)  # Converts to float, handling potential issues with object types
        return torch.tensor(array, dtype=torch.float)

    def process(self):
        self.data_list = []
        # print(f"Starting process. Initial train_names length: {len(self.train_names)}")
        index = 0
        # duration_class = defaultdict(int)
        # midi_class = defaultdict(int)
        # should load musicxml
        for directory in self.train_names:
            print(f"Processing directory {directory}")
            pkl_files = []
            pkl_files.extend(glob.glob(f"{directory}/**/*.pkl", recursive=True))
            pkl_file_path = pkl_files[0]
            xml_files = []
            xml_files.extend(glob.glob(f"{directory}/**/*.xml", recursive=True))
            for xml_file_path in xml_files:
                # print(f"Processing xml {xml_file_path}")
                xml_file_path = Path(xml_file_path)
                XMLDocument = MusicXMLDocument(str(xml_file_path))

                # We have six features: Pitch Name, Midi Number, Octave, Duration, MeaureNumber: should check length of the notes
                notes = XMLDocument.get_notes()

                note_measure_num = [note.measure_number for note in notes]  # Measure Number

                mapped_midi = [midi_map[str(note.pitch[1])] for note in notes]  # Midi Number
                midi_features_list = self.one_hot_convert(mapped_midi, 58)

                mapped_pitch = [pitch_class_map[note.pitch[0]] for note in notes]
                pitch_features_list = self.one_hot_convert(mapped_pitch, 35)  # pitch class: categorical value

                mapped_duration = [duration_map[str(note.note_duration.duration)] for note in notes]  # duration
                duration_features_list = self.one_hot_convert(mapped_duration, 17)
                ## notes information with music21 library
                # max notes
                mapped_position = [i for i in range(len(notes))]
                position_features_list = self.one_hot_convert(mapped_position, 50)  # max number of notes
                try:
                    # Convert warnings to exceptions
                    score = converter.parse(xml_file_path)  # Path object?
                except Warning as w:
                    print("Caught a warning:", w)
                    print(f"file = {directory}")

                # detech warning
                current_offset = 0
                offsetlist = []
                ocativelist = []
                # note_features_list_duration = []
                # print(f"len = {len(score.recurse().notes)}")
                # print(f"len of notes = {len(midi_features_list)}")
                for n in score.recurse().notes:
                    if n.tie is None or n.tie.type == "start":  # not tie
                        ocativelist.append(n.pitch.octave)
                        duration = n.duration.quarterLength
                        current_offset += duration
                        offsetlist.append(current_offset)
                        # note_features_list_duration.append(duration)
                    elif n.tie.type in ["stop", "continue"]:  # only update the offset, not append notes
                        print(f" directory with tie = {directory}")
                        duration = n.duration.quarterLength
                        current_offset += duration
                # print(f"offset = {offsetlist}")
                # assert len(offsetlist) == len(midi_features_list)            
                # assert len(ocativelist) == len(midi_features_list)  
                # normalize offset across whole music score
                # for d in note_features_list_duration:
                #     duration_class[d] += 1

                # min_value = min(offsetlist)
                # range_value = max(offsetlist) - min_value
                # normalized_offsetlist = [(x-min_value)/range_value for x in offsetlist] # normalized offset list

                # We have four edge types: forward, onset, sustain, rest
                notes_graph = score_graph.make_edge(notes)  # edge list: (start note, end note, type of edge)
                onset_edges = []
                voice_edges = []
                forward_edges = []
                slur_edges = []
                sustain_edges = []
                rest_edges = []

                for edge in notes_graph:
                    if edge[2] == 'onset':
                        # print(f"shape = {edge}") # (start, end, type of edges)
                        onset_edges.append(edge[:2])

                    elif edge[2] == 'voice':
                        voice_edges.append(edge[:2])
                    elif edge[2] == 'forward':
                        forward_edges.append(edge[:2])
                    elif edge[2] == 'slur':
                        slur_edges.append(edge[:2])
                    elif edge[2] == 'rest':
                        rest_edges.append(edge[:2])
                    elif edge[2] == 'sustain':
                        sustain_edges.append(edge[:2])

                # midi_features = self.normalize(np.array(midi_features_list))
                # duration_features = self.normalize(np.array(note_features_list_duration))
                # pitch_features = self.normalize(np.array(pitch_features_list))
                # note_measure_num_features = self.normalize(np.array(note_measure_num))
                # normalized_offsetlist = self.normalize(np.array(offsetlist))
                # ocativelist = self.normalize(np.array(ocativelist))
                # encoder = OneHotEncoder(sparse=False)
                # one_hot_encoded_durations = encoder.fit_transform(duration_features)

                midi_features = np.array(midi_features_list)
                duration_features = np.array(duration_features_list)
                # print(f"duration = {set(note_features_list_duration)}")
                pitch_features = np.array(pitch_features_list)
                position_features = np.array(position_features_list)
                # note_measure_num_features = np.array(note_measure_num)
                normalized_offsetlist = np.array(offsetlist)  # not normalized

                ocativelist = np.array(ocativelist)

                pitch_features = self.to_float_tensor(pitch_features)  # categorical
                midi_features = self.to_float_tensor(midi_features)  # categorical
                duration_features = self.to_float_tensor(duration_features)  # categorical
                position_features = self.to_float_tensor(position_features)  # categorical
                # note_measure_num_features = self.to_float_tensor(note_measure_num_features).unsqueeze(1)
                normalized_offsetlist = self.to_float_tensor(normalized_offsetlist).unsqueeze(1)
                ocativelist = self.to_float_tensor(ocativelist).unsqueeze(1)
                # note_features = torch.cat([pitch_features, midi_features, duration_features, note_measure_num_features, normalized_offsetlist, ocativelist], dim=1)
                # note_features = torch.cat([pitch_features, midi_features, duration_features, note_measure_num_features, normalized_offsetlist, ocativelist], dim=1)
                # print(f"pitch feature = {pitch_features.shape}")
                # print(f"midi_features feature = {midi_features.shape}")
                # print(f"midi_features feature = {duration_features.shape}")       
                # print(f"note_measure_num_features feature = {note_measure_num_features.shape}")
                # print(f"normalized_offsetlist feature = {normalized_offsetlist.shape}")
                # print(f"ocativelist feature = {ocativelist.shape}")         
                # note_features = torch.cat([pitch_features, duration_features], dim=1)
                # print(f"shape of features = {midi_features.shape}, {duration_features.shape}, {pitch_features.shape}")
                # print(f"shape of features = {position_features.shape}")
                note_features = torch.cat([pitch_features, midi_features, duration_features, normalized_offsetlist],
                                          dim=1)
                # Stack all the features together to create a single array with all features
                # Each feature becomes a column in the resulting array

                next_edge_index = np.array(forward_edges)
                voice_edge_index = np.array(voice_edges)
                slur_edge_index = np.array(slur_edges)
                onset_edge_index = np.array(onset_edges)
                sustain_edge_index = np.array(sustain_edges)
                rest_edge_index = np.array(rest_edges)

                next_edge_index = torch.tensor(next_edge_index, dtype=torch.long).t().contiguous()
                voice_edge_index = torch.tensor(voice_edge_index, dtype=torch.long).t().contiguous()
                slur_edge_index = torch.tensor(slur_edge_index, dtype=torch.long).t().contiguous()
                onset_edge_index = torch.tensor(onset_edge_index, dtype=torch.long).t().contiguous()
                sustain_edge_index = torch.tensor(sustain_edge_index, dtype=torch.long).t().contiguous()
                rest_edge_index = torch.tensor(rest_edge_index, dtype=torch.long).t().contiguous()

                data1 = HeteroData()
                data1['note'].x = note_features  # node_features is a tensor of shape [num_notes, 4]
                num_nodes = note_features.shape[0]
                edge_types_and_indices = [
                    (('note', 'forward', 'note'), next_edge_index),
                    # (('note', 'voice', 'note'), voice_edge_index),  # Uncomment if needed
                    # (('note', 'slur', 'note'), slur_edge_index),  # Uncomment if needed and available
                    (('note', 'onset', 'note'), onset_edge_index),
                    (('note', 'sustain', 'note'), sustain_edge_index),
                    (('note', 'rest', 'note'), rest_edge_index),
                ]
                # Check and set edge_index for each edge type
                if self.add_self_loops == False:
                    for edge_type, edge_index in edge_types_and_indices:
                        if edge_index.numel() == 0:  # Check if the edge_index tensor is empty
                            # Set to an empty tensor of shape [2, 0] on the same device as node_features
                            # data1[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=node_features.device)
                            continue
                        else:
                            # Directly set the edge_index tensor
                            data1[edge_type].edge_index = edge_index
                            num_edges = data1[edge_type].edge_index.shape[1]
                            edge_weights = torch.ones(num_edges)
                            data1[edge_type].edge_attr = edge_weights
                            # stats[edge_type[1]].add(edge_index.shape[1])
                if self.add_self_loops == True:
                    for edge_type, edge_index in edge_types_and_indices:
                        if edge_index.numel() == 0:  # Check if the edge_index tensor is empty
                            # Set to an empty tensor of shape [2, 0] on the same device as node_features
                            # data1[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=node_features.device)
                            continue
                        else:
                            # Directly set the edge_index tensor
                            num_nodes = data1[edge_type[0]].NUM_NODES  # Adjust based on actual source node type
                            edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
                            data1[edge_type].edge_index = edge_index_with_loops

                            num_edges = data1[edge_type].edge_index.shape[1]
                            edge_weights = torch.ones(num_edges)
                            data1[edge_type].edge_attr = edge_weights

                if self.extract_key_signature(xml_file_path) is not None:
                    key = self.extract_key_signature(xml_file_path)
                    # print(f"key = {key}")
                    key = int(key)  # get the key sigature
                    # since label cannot be negative, label = label + 7
                    key = key + 7
                    if key in range(0, 15):  # should be valid fifth
                        # print(f"key = {key} is in correct range")
                        data1.y = torch.tensor([key], dtype=torch.long)
                        if self.transform:
                            data1 = self.transform(data1)

                            # print(data1['note'].x.shape[0])
                            # final_size = data1['note'].x.shape[0]
                        # replace suffix

                        try:
                            # Placeholder for Sck analysis, could be numpy
                            with open(pkl_file_path, 'rb') as file:
                                clusters = pickle.load(file)
                            # print(f"len of cluster = {len(clusters)}")
                            cluster_tuple = ()
                            assert clusters[0].shape[0] == len(notes)

                            for cluster in clusters:
                                # print(f"cluster is = {cluster}")
                                cur_cluster = torch.tensor(cluster, dtype=torch.float32)
                                # if self.transform:
                                #     cur_cluster = self.pad_cluster(final_size, cur_cluster)
                                cluster_tuple = cluster_tuple + (cur_cluster,)
                        except:
                            print(f"len of notes = {len(notes)}, clusters[0].shape[0] = {clusters[0].shape[0]}")
                            print(f"File_path = {xml_file_path} cannot be loaded")

                    else:
                        print(f"key fifths not in the range of [-7,7] {xml_file_path}")
                        continue

                        # also need to pad cluster matrix

                    data_dict = {}
                    data_dict["name"] = str(xml_file_path).removesuffix('.xml')
                    data_dict["data"] = data1
                    data_dict["cluster"] = cluster_tuple
                    # print(f"sample cluster tuple = {cluster_tuple[0].shape}, {cluster_tuple[1].shape}")
                    torch.save(data_dict, os.path.join(self.processed_dir, f'{index}_processed.pt'))
                    self.data_list.append(data_dict)
                    # print(f"Appending to data_list. Current length before append: {len(self.data_list)}")
                index = index + 1
        # print(f"midi class = {midi_class.keys()}")
        # print(f"duration class = {duration_class.keys()}")
        # with open("midi_map", 'w') as file:
        #     json.dump(midi_class, file)
        # with open("duration_map", 'w') as file:
        #     json.dump(duration_class, file)

    def extract_key_signature(self, file_path):
        # Load and parse the MusicXML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Search for the 'key' element
        for part in root.findall('part'):
            for measure in part.findall('measure'):
                attributes = measure.find('attributes')
                if attributes is not None:
                    key = attributes.find('key')
                    if key is not None:
                        key_fifths = key.find('fifths').text if key.find('fifths') is not None else "Unknown"
                        return key_fifths
        return None

    def hetero_to_networkx(self, obj_idx):
        # Define a color map for edge types

        edge_type_color = {
            'forward': 'red',
            'onset': 'green',
            'sustain': 'blue',
            'rest': 'yellow',
        }

        file_name = self.processed_file_names[obj_idx]
        file_path = os.path.join(self.processed_dir, file_name)
        hetero_data = torch.load(file_path)
        G = nx.DiGraph()  # Directed graph to accommodate directed edges, if needed

        # Add nodes
        for node_type in hetero_data.node_types:
            for node_id in range(hetero_data[node_type].NUM_NODES):
                G.add_node(f"{node_type}_{node_id}", type=node_type)

        # Add edges with colors
        for edge_type in hetero_data.edge_types:
            color = edge_type_color.get(edge_type[1], 'black')  # Default to black if type not found
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
    #
    # transform = Pad(max_num_nodes, max_num_edges, node_padding,  edge_padding)
    #
    #     dataset = HeterGraph(root ="processed/heterdata/", train_names = train_names, transform = transform, add_pad_mask =True)
    #
    #     dataset = HomoGraph(root ="processed/homodata/", train_names = train_names, transform = transform)
    #
    #     print(f"data1 {dataset[0]}")
    train_loader = DataLoader(dataset, batch_size=1)

    for data, sck_cluster_tuple in train_loader:
        print(f"data = {data}, {sck_cluster_tuple[0][0].shape}, {sck_cluster_tuple[1][0].shape}")

#
#
#     node_padding = NodeTypePadding({
#         'note': 0.0,
#     }, default=1.0)
#
#     edge_padding = EdgeTypePadding({
#         ('note', 'forward', 'note'): 0,
#         ('note', 'onset', 'note'):0,
#         ('note', 'sustain', 'note'):0,
#         ('note', 'rest', 'note'):0
#     }, default=1.5)
#
#
#     max_num_nodes = {'note':300}
#     max_num_edges = {
#         ('note', 'forward', 'note'): 200,
#         ('note', 'onset', 'note'):200,
#         ('note', 'sustain', 'note'):200,
#         ('note', 'rest', 'note'):200
#     }
#
#     hetero_graph = HeterGraph(root='path_to_data', train_names=train_names)
#     data1 = hetero_graph[0]
#     first_5_name = train_names[:5]
#     print(f"first_5_name = {first_5_name}")
#     for i in range(5):
#         file_path = first_5_name[i]
#         parts = file_path.split('/')
#         extracted_path = '_'.join(parts[-2:])  # Join the second to last and last part with an underscore
#         extracted_path = extracted_path.split('.')[0]  # Remove the file extension
#         G = hetero_graph.hetero_to_networkx(i)
#
#         edge_colors = nx.get_edge_attributes(G, 'color').values()
#
#         plt.figure(figsize=(12, 8))
#         pos = nx.spring_layout(G)  # Or any other layout
#         nx.draw(G, pos, edge_color=edge_colors, with_labels=True, node_size=700, node_color="lightblue", arrows=True)
#         plt.savefig(f"{extracted_path}_colored.jpg")
