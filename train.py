import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from utils import prepare_data_loaders, prepare_model
from config import *

torch.manual_seed(45)


def debug_matrices(grouping_matrix_true, cluster_matrix_pred, grouping_matrix_pred):
    num_decimals = 3
    grouping_matrix_true_round = np.round(grouping_matrix_true.cpu().detach().numpy(), decimals=num_decimals)
    grouping_matrix_pred_after = np.round(torch.matmul(cluster_matrix_pred, cluster_matrix_pred.t()).cpu().detach().numpy(),
                                          decimals=num_decimals)
    grouping_matrix_pred_round = np.round(grouping_matrix_pred.cpu().detach().numpy(), decimals=num_decimals)
    return grouping_matrix_true_round, grouping_matrix_pred_after, grouping_matrix_pred_round


def train_loop(model, train_loader):
    train_losses = []
    for count, databatch in tqdm(enumerate(train_loader)):
        filename = Path(databatch['name'][0])
        data = databatch['data']
        data = data.to(DEVICE)

        cluster_matrices_true = [c[0] for c in databatch['cluster']]
        grouping_matrices_true = [
            torch.linalg.multi_dot(cluster_matrices_true[:i+1])
            if i > 0
            else cluster_matrices_true[0]
            for i, _ in enumerate(cluster_matrices_true)
        ]
        grouping_matrices_true = [
            torch.matmul(cluster_matrix_true, cluster_matrix_true.t())
            for cluster_matrix_true in grouping_matrices_true
        ]

        final_embedding, cluster_results = model(data, grouping_matrices_true)

        # atrue, aafter, apred = debug_matrices(grouping_matrix_true, cluster_matrix_1_pred, grouping_matrix_1_pred)

        train_loss = torch.sum(torch.stack(cluster_results['grouping_losses']))
        train_losses.append(train_loss.item())
        train_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return np.mean(train_losses)


def validation_loop(model, valid_loader):
    model.eval()
    with torch.no_grad():
        val_losses = []
        for count, databatch in enumerate(valid_loader):
            filename = Path(databatch['name'][0])
            data = databatch['data']

            cluster_matrices_true = [c[0] for c in databatch['cluster']]
            grouping_matrices_true = [cluster_matrices_true[0]]
            for i, cluster_matrix_true in enumerate(cluster_matrices_true):
                if i == 0: continue
                grouping_matrices_true.append(torch.matmul(grouping_matrices_true[i - 1], cluster_matrices_true[i]))
            grouping_matrices_true = [
                torch.matmul(cluster_matrix_true, cluster_matrix_true.t())
                for cluster_matrix_true in grouping_matrices_true
            ]

            data = data.to(DEVICE)
            final_emb, cluster_results = model(data, grouping_matrices_true)

            val_loss = torch.sum(torch.stack(cluster_results['grouping_losses']))
            val_losses.append(val_loss)

    return np.mean(val_losses)


if __name__ == "__main__":
    from model.schenker_GNN_model import GroupMat
    from data_processing import HeterGraph
    train_loader, valid_loader = prepare_data_loaders(TRAIN_NAMES, SAVE_FOLDER, HeterGraph)

    model, optimizer, scheduler = prepare_model(
        NUM_FEAT,
        EMB_DIM,
        HIDDEN_DIM,
        NUM_CLUSTERING_LAYERS,
        model_class=GroupMat,
        device=DEVICE
    )
    model.to(DEVICE)

    train_loss_curve = []
    valid_loss_curve = []
    train_acc_curve = []
    valid_acc_curve = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}:")
        train_loss = train_loop(model, train_loader)
        print(f'Training Loss: {train_loss:.4f}')
        valid_loss = validation_loop(model, valid_loader)
        print(f'Validation Loss: {valid_loss:.4f}')
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
