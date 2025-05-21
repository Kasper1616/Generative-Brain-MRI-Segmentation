import pyro


num_epochs = 5
test_frequency = 10
lr = 1e-3


def train(svi, train_loader, device):
    # initialize loss accumulator
    epoch_loss = 0.

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
    # initialize loss accumulator
    test_loss = 0.
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



def run_model(vae, svi, train_loader, test_loader, num_epochs=num_epochs, test_frequency=test_frequency, lr=lr, device="cpu"):
    pyro.clear_param_store()  # clear everything before instantiation
    vae.to(device)

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(num_epochs):
        total_epoch_loss_train = train(svi, train_loader, device)
        train_elbo.append(-total_epoch_loss_train)
        print(f"[Epoch {epoch+1}]")
        print("Mean train loss: %.4f" %  total_epoch_loss_train)

        if epoch % test_frequency == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, device)
            test_elbo.append(-total_epoch_loss_test)
            print("Mean test loss: %.4f" % total_epoch_loss_test)
        print("")
    return vae