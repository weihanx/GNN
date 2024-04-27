import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt

from utils import start_logger, prepare_data_loaders, prepare_model
import Sckgroupmat
import dataset_heter
from config import *

torch.manual_seed(42)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x_dict, data.edge_index_dict, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


def check_NaN(final_emb, filename, optimizer):
    if torch.isnan(final_emb["note"][0][0]):
        print(f"PROBLEM: {filename}")
        # raise ValueError()
    else:
        print(f"NOT A PROBLEM: {filename}")


def train_sck(model, loss_fn, train_loader, class_weight):
    logging.debug(f"Training...")
    train_loss = []
    count = 0
    correct = 0
    for databatch in train_loader:

        optimizer.zero_grad()

        count = count + 1
        filename = Path(databatch['name'][0])
        # if str(filename) != "schenkerian_clusters\\Septimi_6_trans_c#\\Septimi_6_trans_c#":
        #     continue
        data = databatch['data']
        sck_mat_tuple = databatch['cluster']
        data = data.to(DEVICE)
        final_emb, s1_pred = model(data)

        # check_NaN(final_emb, filename, optimizer)

        filename.parent.mkdir(parents=True, exist_ok=True)

        cluster1 = sck_mat_tuple[0][0]
        cluster2 = sck_mat_tuple[1][0]
        s1 = torch.matmul(cluster1, cluster1.t())
        s2 = torch.matmul(cluster1, cluster2)
        s2 = torch.matmul(s2, s2.t())
        s1 = s1.to(DEVICE)
        s2 = s2.to(DEVICE)
        loss2 = loss_fn(s1_pred.float(), s1.float())
        # loss3 = loss_fn(s2_pred.float(), s2.float())
        # loss = class_weight[0] * loss2 + class_weight[1] * loss3
        train_loss.append(loss2.item())
        loss2.backward()

        optimizer.step()
        # _, predicted_labels = torch.max(out, 1)

        # correct += (predicted_labels == data.y).sum().item()
        # train_acc = correct / count
    return np.mean(train_loss)  # average loss for this epoch


def validate_sck(model, loss_fn, valid_loader, class_weight):
    logging.debug(f"Validating...")
    model.eval()
    with torch.no_grad():
        correct = 0
        val_loss = []
        count = 0
        for databatch in valid_loader:
            filename = Path(databatch['name'][0])
            data = databatch['data']
            sck_mat_tuple = databatch['cluster']
            data = data.to(DEVICE)
            final_emb, s1_pred = model(data)
            # try:
            #     check_NaN(final_emb, filename, optimizer)
            # except ValueError as e:
            #     break
            true_label = data.y
            filename.parent.mkdir(parents=True, exist_ok=True)

            cluster1 = sck_mat_tuple[0][0]
            cluster2 = sck_mat_tuple[1][0]
            s1 = torch.matmul(cluster1, cluster1.t())
            s2 = torch.matmul(cluster1, cluster2)
            s2 = torch.matmul(s2, s2.t())
            s1 = s1.to(DEVICE)
            s2 = s2.to(DEVICE)
            loss2 = loss_fn(s1_pred.float(), s1.float())
            # loss3 = loss_fn(s2_pred.float(), s2.float())
            # loss = class_weight[0] * loss2 + class_weight[1] * loss3
            count += 1
            val_loss.append(loss2.item())
            # _, predicted_labels = torch.max(out, 1)

            # correct += (predicted_labels == data.y).sum().item()

        # val_acc = correct / count
    return np.mean(val_loss)


if __name__ == "__main__":
    start_logger()

    TRAIN_NAMES = "train-names.txt"
    SAVE_FOLDER = "hetergraph0422_4feature/"
    NUM_FEAT = 111
    EMB_DIM = 32
    HIDDEN_DIM = 200
    NUM_CLASS = 15
    dataset_class = dataset_heter.HeterGraph
    train_loader, valid_loader = prepare_data_loaders(TRAIN_NAMES, SAVE_FOLDER, dataset_class)

    model, optimizer, scheduler = prepare_model(
        NUM_FEAT,
        EMB_DIM,
        HIDDEN_DIM,
        NUM_CLASS,
        model_class=Sckgroupmat.GroupMat,
        device=DEVICE
    )

    model.to(DEVICE)
    # Training and validation loop
    num_epochs = 50
    train_loss_curve = []
    valid_loss_curve = []
    train_acc_curve = []
    valid_acc_curve = []
    loss_fn = SIM_CRITERION
    class_weight = [0.7, 0.3]

    for epoch in range(num_epochs):
        train_loss = train_sck(model, loss_fn, train_loader, class_weight)
        valid_loss = validate_sck(model, loss_fn, valid_loader, class_weight)
        print(f'Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
        scheduler.step()
        train_loss_curve.append(train_loss)
        valid_loss_curve.append(valid_loss)
    logging.info(f"Plotting loss and acc curve")
    epochs = np.arange(0, 50, dtype=int)

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('Loss Curve and Acc Curve', fontsize=16, y=1.02)

    # Loss curves
    axs[0].plot(epochs, train_loss_curve, label='Training Loss')
    axs[0].plot(epochs, valid_loss_curve, label='Validation Loss')
    axs[0].set_xlabel('Epoch Number')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss Curves')
    axs[0].legend()

    # Accuracy curves
    axs[1].plot(epochs, train_acc_curve, label='Training Accuracy')
    axs[1].plot(epochs, valid_acc_curve, label='Validation Accuracy')
    axs[1].set_xlabel('Epoch Number')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy Curves')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("SAGE_bsz1.png")
