import torch

def train_model(model, epochs, train_loader, test_loader, criterion, optimizer, device):
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []

    for epoch in range(epochs):
        # training
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # calculate the loss and accuracies
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # testing per epoch 
        model.eval()
        test_running_loss = 0.0
        test_correct, test_total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # forward pass (no backward pass for testing)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        epoch_test_loss = test_running_loss / len(test_loader)
        epoch_test_acc = 100 * test_correct / test_total
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)

        print(f"Epoch {epoch + 1} out of {epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}%")
        print(f"Test Loss: {epoch_test_loss:.4f} - Test Acc: {epoch_test_acc:.2f}%")

    return train_loss, train_acc, test_loss, test_acc


def test_model(model, epochs, test_loader, criterion, device):
    # test loss
    test_loss = []
    # test accuracy
    test_acc = []
    # copied from my train as I put test per epoch in the train
    # just testing here

    for epoch in range(epochs):
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        epoch_test_loss = test_running_loss / len(test_loader)
        epoch_test_acc = 100 * test_correct / test_total
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)
    return test_loss, test_acc

