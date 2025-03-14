import torch

def train_model_vae(data_loader, model, criterion, optimizer, nepochs, device):
    #List to store loss to visualize
    train_losses = []
    train_acc = []
    start_epoch = 0

    for epoch in range(start_epoch, nepochs):
        train_loss = 0.
        valid_loss = 0.
        correct = 0

        model.train()
        for batch_idx, (input_, target) in enumerate(data_loader):
            input_ = input_.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            output_, mean, logvar = model(input_)

            # calculate the batch loss
            loss = criterion(input_, output_, mean, logvar)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item() * input_.size(0)

        # calculate average losses
        train_loss = train_loss/len(data_loader.dataset)
        train_losses.append(train_loss)

        # print losses statistics 
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss))

    return train_losses, train_acc
