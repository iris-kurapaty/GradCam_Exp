from torch_lr_finder import LRFinder
import torch.optim as optim
import torch.nn as nn


def get_lr_value(model, train_loader):
    optimizer = optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state

    return optimizer, criterion

