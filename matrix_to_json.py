import numpy as np
import pickle
import json


def read_file(file_path):
    with open(file_path, 'rb') as file:
        matrices = pickle.load(file)
    return matrices


def update_depths(matrices):
    n_rows = matrices[0].shape[0]
    cluster_to_index = {i: i for i in range(n_rows)}
    depths = [0] * n_rows

    for matrix in matrices:
        for c in range(matrix.shape[1]):
            rows_with_ones = np.where(matrix[:, c] == 1)[0]
            if len(rows_with_ones) > 0:
                r = rows_with_ones[0]
                note_index = cluster_to_index[r]
                depths[note_index] += 1
                del cluster_to_index[r]
                cluster_to_index[c] = note_index

    return depths


def convert_to_format(depths, global_to_depths):
    n_verticalities = max(value[1] for value in global_to_depths.values()) + 1
    depths_format = [[-1] * n_verticalities for _ in range(4)]
    for i in range(len(depths)):
        voice, verticality = global_to_depths[i]
        depths_format[voice][verticality] = depths[i]
    return depths_format


def process_raw(filepath):
    with open(filepath, "r") as file:
        data_dict = json.load(file)
        t = data_dict["trebleNotes"]
        b = data_dict["bassNotes"]
        ti = data_dict["innerTrebleNotes"]
        bi = data_dict["innerBassNotes"]
        idx = 0
        global_to_depths = {}
        for i in range(len(t["pitchNames"])):
            if t["pitchNames"][i] != "_":
                global_to_depths[idx] = (0, i)
                idx += 1
            if b["pitchNames"][i] != "_" and b["pitchNames"][i] != t["pitchNames"][i]:
                global_to_depths[idx] = (3, i)
                idx += 1
            if ti["pitchNames"][i] != "_" and ti["pitchNames"][i] != t["pitchNames"][i] and ti["pitchNames"][i] != \
                    b["pitchNames"][i]:
                global_to_depths[idx] = (1, i)
                idx += 1
            if bi["pitchNames"][i] != "_" and bi["pitchNames"][i] != t["pitchNames"] and bi["pitchNames"][i] != \
                    b["pitchNames"][i] and bi["pitchNames"][i] != ti["pitchNames"][i]:
                global_to_depths[idx] = (2, i)
                idx += 1
    return global_to_depths


def generate_paths(piece):
    base_path = 'schenkerian_clusters'
    ground_truth_path = f'{base_path}/{piece}/{piece}.json'
    file_path = f'{base_path}/{piece}/{piece}.pkl'
    return ground_truth_path, file_path


def update_json_depths(ground_truth_path, depths_format):
    with open(ground_truth_path, 'r') as file:
        data_dict = json.load(file)
    
    data_dict["trebleNotes"]["depths"] = depths_format[0]
    data_dict["innerTrebleNotes"]["depths"] = depths_format[1]
    data_dict["innerBassNotes"]["depths"] = depths_format[2]
    data_dict["bassNotes"]["depths"] = depths_format[3]
    
    new_path = ground_truth_path.replace('.json', '_predict.json')
    with open(new_path, 'w') as file:
        json.dump(data_dict, file, indent=4)
    return


def matrix_to_json(piece):
    ground_truth_path, file_path = generate_paths(piece)
    global_to_positions = process_raw(ground_truth_path)
    matrices = read_file(file_path)
    depths = update_depths(matrices)
    depths_format = convert_to_format(depths, global_to_positions)
    update_json_depths(ground_truth_path, depths_format)
    return


if __name__ == "__main__":
    piece = 'Primi_3'
    matrix_to_json(piece)


    #ground_truth_path, file_path = generate_paths(piece)
    #global_to_positions = process_raw(ground_truth_path)
    #t_depth, b_depth, ti_depth, bi_depth, global_to_positions = process_raw(ground_truth_path)
    #print('\n')
    #print("--------------------Ground Truth--------------------")
    #print(t_depth)
    #print(ti_depth)
    #print(bi_depth)
    #print(b_depth)
    #print('\n')
    #print("--------------------Generated-----------------------")
    #matrices = read_file(file_path)
    #depths = update_depths(matrices)
    #depths_format = convert_to_format(depths, global_to_positions)
    #new_json_path = update_json_depths(ground_truth_path, depths_format)
    #for i in range(len(depths_format)):
    #    print(depths_format[i])