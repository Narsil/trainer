import torch
import torch.nn.functional as F
from torch import nn
from trainer import EpochTrainer
from torchvision import datasets, transforms


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MyTrainer(EpochTrainer):
    EPOCHS_DEFAULT = 100
    EXP_NAME_DEFAULT = "mnist"

    def get_loss(self, model, batch):
        inputs = batch[0]
        targets = batch[1]
        output = model(inputs)
        loss = F.nll_loss(output, targets)
        return loss, {}


def main():
    lr = 1e-4

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0, max_lr=lr, cycle_momentum=False
    )

    dataset = (
        datasets.MNIST(
            "../data",
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
    )

    trainer = MyTrainer(model, dataset, optimizer, scheduler=scheduler, **vars(args))
    trainer.train()


if __name__ == "__main__":
    main()
