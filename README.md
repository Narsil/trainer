Simple Pytorch Trainer
========================


Installation
------------

`pip install -e git://github.com/Narsil/trainer/`


Usage
-----


```python
from trainer import EpochTrainer

class MyTrainer(EpochTrainer):
    def get_loss(self, model, batch):
        ....
        loss = ...
        losses = {'aux_loss': aux_loss, ....}
        return loss, losses

model = MyModel()
dataset = MyDataset()
optimizer = MyOptimizer()

trainer = MyTrainer(model, dataset, optimizer)
trainer.train()
```

AdvancedUsage
-----


You can override use command line arguments or defaults to change arguments of the main
EpochTrainer

```python
from trainer import EpochTrainer

class MyTrainer(EpochTrainer):
    EPOCHS = 100  # More epochs than the default
    def get_loss(self, model, batch):
        ....
        loss = ...
        losses = {'aux_loss': aux_loss, ....}
        return loss, losses

model = MyModel()
dataset = MyDataset()
optimizer = MyOptimizer()


parser = argparse.ArgumentParser(description="My awesome command")
# Add Trainer arguments
parser = MyTrainer.parser(parser)
parser.add_argument(
    "--lr",
    type=float,
    default=LEARNING_RATE_DEFAULT,
    help=f"(default {LEARNING_RATE_DEFAULT})",
)

trainer = MyTrainer(model, dataset, optimizer)
trainer.train()
```

And now you can override all default arguments from trainer with the command line, and you changed the epochs default to 100 to better
suite you in your training.
