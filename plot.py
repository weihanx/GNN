# plot with package: https://github.com/YingfanWang/PaCMAP
import pacmap
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path, data_prefix, label_prefix):
    np_list, y_list = [], []
    with open(file_path, "r") as file:
        file_content = [line.strip() for line in file.readlines()]
    
    for fc in file_content:
        cur_x = np.load(fc)
        y_path = fc.replace(data_prefix, label_prefix)
        cur_y = np.load(y_path)
        np_list.append(cur_x)
        y_list.append(cur_y)
    
    X = np.vstack(np_list)
    y = np.vstack(y_list)
    
    return X, y

def visualize_with_pacmap(X, y, save_path, n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0):
    embedding = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio) 
    X_transformed = embedding.fit_transform(X)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y, s=0.6)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.savefig(save_path, dpi=300)


X, y = load_data("nosckout.txt", "training_out_nosck", "true_train_label")
visualize_with_pacmap(X, y, './fourfeat/trainnosck.png')


X, y = load_data("sckout.txt", "training_sck_out", "true_train_label")
visualize_with_pacmap(X, y, './fourfeat/trainsck.png')

X, y = load_data("weightedsckout.txt", "training_sck_out_weighted", "true_train_label_weighted")
visualize_with_pacmap(X, y, './fourfeat/weightedtrainsck.png')


X, y = load_data("valid_nosckout.txt", "valid_nosck_out", "true_valid_label")
visualize_with_pacmap(X, y, './fourfeat/nosckvalid.png')

X, y = load_data("valid_sckout.txt", "valid_sck_out", "true_valid_label")
visualize_with_pacmap(X, y, './fourfeat/validsck.png')

X, y = load_data("weighted_valid_sckout.txt", "valid_sck_out_weighted", "true_valid_label_weighted")
visualize_with_pacmap(X, y, './fourfeat/weightedvalidsck.png')
