from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np
import pyro
import torch


num_epochs = 5
test_frequency = 10
lr = 1e-3


def train(svi, train_loader, device):
    """
    Train the model for one epoch.
    :param svi: SVI object
    :param train_loader: training data loader
    :param device: device to run the model on
    """
    # initialize loss accumulator
    epoch_loss = 0.0

    for img, mask in train_loader:
        # move data to device
        img = img.to(device)
        mask = mask.to(device)

        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(img, mask)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader, device):
    """
    Evaluate the model on the test set.
    :param svi: SVI object
    :param test_loader: test data loader
    :param device: device to run the model on
    """
    # initialize loss accumulator
    test_loss = 0.0
    # compute the loss over the entire test set
    for img, mask in test_loader:
        # move data to device
        img = img.to(device)
        mask = mask.to(device)

        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(img, mask)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def run_model(
    vae,
    svi,
    train_loader,
    test_loader,
    num_epochs=num_epochs,
    test_frequency=test_frequency,
    device="cpu",
):
    """
    Run the model with the given parameters.
    :param vae: VAE model
    :param svi: SVI object
    :param train_loader: training data loader
    :param test_loader: test data loader
    :param num_epochs: number of epochs to train
    :param test_frequency: frequency of testing
    :param lr: learning rate
    :param device: device to run the model on
    :return: trained VAE model
    """
    pyro.clear_param_store()  # clear everything before instantiation
    vae.to(device)

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(num_epochs):
        total_epoch_loss_train = train(svi, train_loader, device)
        train_elbo.append(-total_epoch_loss_train)
        print(f"[Epoch {epoch + 1}]")
        print("Mean train loss: %.4f" % total_epoch_loss_train)

        if epoch % test_frequency == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, device)
            test_elbo.append(-total_epoch_loss_test)
            print("Mean test loss: %.4f" % total_epoch_loss_test)
        print("")
    return vae


def bce_loss(recon_x, x):
    """
    Binary Cross Entropy loss
    :param recon_x: reconstructed image
    :param x: original image
    :return: BCE loss
    """
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction="mean")
    return BCE


def dice_score(recon_x, x):
    """
    DICE score
    :param recon_x: reconstructed image
    :param x: original image
    :return: DICE loss
    """
    smooth = 1e-6
    intersection = (recon_x * x).sum()
    return (2.0 * intersection + smooth) / (recon_x.sum() + x.sum() + smooth)


def trace_elbo(model, data):
    """
    Trace the ELBO for a given model and data.
    :param model: VAE model
    :param data: data to evaluate
    :return: ELBO
    """
    elbo = pyro.infer.Trace_ELBO()
    return elbo.differentiable_loss(model, data)


def print_losses(test_loader, model, device):
    """
    Print the losses for the test set.
    :param test_loader: test data loader
    :param model: VAE model
    :param device: device to run the model on
    """
    model.eval() 
    with torch.no_grad():
        # print bce loss
        print("BCE loss:")
        total_bce = 0.0
        n_batches = 0
        for img, mask in test_loader:
            # move data to device
            img = img.to(device)
            mask = mask.to(device)

            # Check if model is a VAE or a regular model
            if hasattr(model, "model"):
                # VAE model
                pred_mask = model.model(img)
            else:
                # regular model
                pred_mask = model(img)
            total_bce += bce_loss(pred_mask, mask)
            n_batches += 1
        avg_bce = total_bce / n_batches
        print(np.round(avg_bce.item(), 2))
        print("")

        # print average Dice score for the whole test set
        print("Average DICE score:")
        total_dice = 0.0
        n_samples = 0
        for img, mask in test_loader:
            # move data to device
            img = img.to(device)
            mask = mask.to(device)
            # Ensure mask is binary
            mask = (mask > 0.5).float()
            # Check if model is a VAE or a regular model
            if hasattr(model, "model"):
                # VAE model
                pred_mask = model.model(img)
            else:
                # regular model
                pred_mask = model(img)
            # threshold predictions to obtain binary mask
            pred_mask = (pred_mask > 0.5).float()
            # Compute dice per sample
            for i in range(img.size(0)):
                dice = dice_score(pred_mask[i:i+1], mask[i:i+1])
                total_dice += dice.item()
                n_samples += 1
        avg_dice = total_dice / n_samples
        print(np.round(avg_dice, 2))
        print("")

        # print elbo loss if available
        if hasattr(model, "model"):
            print("ELBO loss:")
            optimizer = Adam({"lr": lr})
            svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
            print(np.round(evaluate(svi, test_loader, device), 2))
        else:
            print("No ELBO loss available.")
