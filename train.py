import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import torch

from utils import prepare_data_loaders, prepare_model
from config import *

# torch.manual_seed(42)


def debug_matrices(grouping_matrix_true, cluster_matrix_pred, grouping_matrix_pred):
    num_decimals = 2
    grouping_matrix_true_round = np.round(grouping_matrix_true.detach().numpy(), decimals=num_decimals)
    grouping_matrix_pred_after = np.round(torch.matmul(cluster_matrix_pred, cluster_matrix_pred.t()).detach().numpy(),
                                          decimals=num_decimals)
    grouping_matrix_pred_round = np.round(grouping_matrix_pred.detach().numpy(), decimals=num_decimals)
    return grouping_matrix_true_round, grouping_matrix_pred_after, grouping_matrix_pred_round


def train_loop(model, train_loader):
    logging.debug(f"Training...")
    train_loss = []
    for count, databatch in enumerate(train_loader):

        filename = Path(databatch['name'][0])
        data = databatch['data']
        data = data.to(DEVICE)

        cluster_matrix_true = databatch['cluster'][0][0]
        grouping_matrix_true = torch.matmul(cluster_matrix_true, cluster_matrix_true.t())

        final_embedding, cluster_matrix_pred, grouping_loss, grouping_matrix_pred = model(data, grouping_matrix_true)

        atrue, aafter, apred = debug_matrices(grouping_matrix_true, cluster_matrix_pred, grouping_matrix_pred)

        train_loss.append(grouping_loss.item())
        grouping_loss.backward()

        if count % PRINT_EVERY == 0 and PRINT_LOSS:
            print(filename)
            print(grouping_loss.item())
        if count % PRINT_EVERY == 0 and PRINT_MATRICES:
            print(filename)
            print(grouping_matrix_pred)
            print(grouping_matrix_true)

        optimizer.step()
        optimizer.zero_grad()

    return np.mean(train_loss)


def validation_loop(model, valid_loader):
    logging.debug(f"Validating...")
    model.eval()
    with torch.no_grad():
        val_loss = []
        for count, databatch in enumerate(valid_loader):
            filename = Path(databatch['name'][0])
            data = databatch['data']

            cluster_matrix_true = databatch['cluster'][0][0]
            grouping_matrix_true = torch.matmul(cluster_matrix_true, cluster_matrix_true.t())

            data = data.to(DEVICE)
            final_emb, cluster_matrix_pred, grouping_loss, grouping_matrix_pred = model(data, grouping_matrix_true)

            filename.parent.mkdir(parents=True, exist_ok=True)

            val_loss.append(grouping_loss.item())
            if count % PRINT_EVERY == 0 and PRINT_MATRICES:
                print(filename)
                print(grouping_matrix_pred)
                print(grouping_matrix_true)

    return np.mean(val_loss)


if __name__ == "__main__":
    from model.schenker_GNN_model import GroupMat
    from dataset_heter import HeterGraph
    train_loader, valid_loader = prepare_data_loaders(TRAIN_NAMES, SAVE_FOLDER, HeterGraph)

    model, optimizer, scheduler = prepare_model(
        NUM_FEAT,
        EMB_DIM,
        HIDDEN_DIM,
        NUM_CLASS,
        model_class=GroupMat,
        device=DEVICE
    )
    model.to(DEVICE)

    train_loss_curve = []
    valid_loss_curve = []
    train_acc_curve = []
    valid_acc_curve = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_loop(model, train_loader)
        valid_loss = validation_loop(model, valid_loader)
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
