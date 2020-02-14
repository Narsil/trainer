import tqdm
import torch
import logging
import datetime
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__file__)


class Trainer:
    BATCH_SIZE_DEFAULT = 32
    GLOBAL_STEP_DEFAULT = 0
    BEST_LOSS_DEFAULT = float("inf")
    EXP_NAME_DEFAULT = "default"
    MAX_PATIENCE_DEFAULT = 2
    NUM_WORKERS_DEFAULT = 2
    BEST_MODEL_FILENAME_DEFAULT = "best_model.pth"
    TRAIN_PROPORTION_DEFAULT = 0.8
    CHECKPOINT_INTERVAL_DEFAULT = datetime.timedelta(minutes=30)

    def __init__(self, model, dataset, optimizer, **kwargs):
        if torch.cuda.is_available():
            model = model.cuda()
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer

        self.patience = 0
        self.last_checkpoint = datetime.datetime.now()

        # Overrides
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.train_loader, self.test_loader = self.split_dataset(dataset)
        self.logdir = self.get_logdir()

        # We log to tensorboard
        self.writer = SummaryWriter(self.logdir)

    def log_configuration(self):
        logger.debug("Configuration : ")
        for name, value in self.__dict__.items():
            logger.debug(f"{name}: {value}")

    def get_logdir(self):
        return (
            f"runs/{self.exp_name}/{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
        )

    @classmethod
    def parser(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(description="Trainer default arguments.")

        parser.add_argument(
            "--num-workers",
            type=int,
            default=cls.NUM_WORKERS_DEFAULT,
            help=f"(default {cls.NUM_WORKERS_DEFAULT})",
        )
        parser.add_argument(
            "--max-patience",
            type=int,
            default=cls.MAX_PATIENCE_DEFAULT,
            help=f"(default {cls.MAX_PATIENCE_DEFAULT})",
        )
        parser.add_argument(
            "--best-loss",
            type=int,
            default=cls.BEST_LOSS_DEFAULT,
            help=f"Don't log any loss above that value (default {cls.BEST_LOSS_DEFAULT})",
        )
        parser.add_argument(
            "--train_proportion",
            type=float,
            default=cls.TRAIN_PROPORTION_DEFAULT,
            help=f"(default {cls.TRAIN_PROPORTION_DEFAULT})",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=cls.BATCH_SIZE_DEFAULT,
            help=f"(default {cls.BATCH_SIZE_DEFAULT})",
        )
        parser.add_argument(
            "--global_step",
            type=int,
            default=cls.GLOBAL_STEP_DEFAULT,
            help=f"(default {cls.GLOBAL_STEP_DEFAULT})",
        )
        parser.add_argument(
            "--now",
            type=datetime.datetime,
            default=datetime.datetime.now(),
            help=f"(default: now)",
        )
        parser.add_argument(
            "--exp_name",
            type=str,
            default=cls.EXP_NAME_DEFAULT,
            help=f"(default {cls.EXP_NAME_DEFAULT})",
        )
        parser.add_argument(
            "--best-model-filename",
            type=str,
            default=cls.BEST_MODEL_FILENAME_DEFAULT,
            help=f"(default {cls.BEST_MODEL_FILENAME_DEFAULT})",
        )
        parser.add_argument(
            "--checkpoint-interval",
            type=datetime.timedelta,
            default=cls.CHECKPOINT_INTERVAL_DEFAULT,
            help=f"Interval in minutes (default {cls.CHECKPOINT_INTERVAL_DEFAULT})",
        )

        return parser

    def split_dataset(self, dataset):
        raise NotImplementedError(
            "split_dataset should be implemented by Trainer subclasses"
        )

    def to_device(self, model, batch):
        device = next(model.parameters()).device
        return self._to_device(batch, device)

    def _to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, (list, tuple)):
            return [self._to_device(el, device) for el in batch]
        elif isinstance(batch, dict):
            return {k: self._to_device(v, device) for k, v in batch.items()}
        else:
            raise NotImplementedError("We don't know how to move this batch to device")

    def get_loss(self, model, batch, reduction="mean"):
        raise NotImplementedError("Please implement your own loss !")

    def end_test(self, total_loss, total, losses, model):
        total_loss /= total
        self.writer.add_scalar("test/loss", total_loss, global_step=self.global_step)
        for loss_name, value in losses.items():
            value /= total
            self.writer.add_scalar(
                f"test/{loss_name}", value, global_step=self.global_step
            )

        if datetime.datetime.now() - self.last_checkpoint > self.checkpoint_interval:
            filename = f"{self.logdir}/model_{self.epoch:03d}.pth"
            logger.debug(f"Saving checkpoint {filename})")
            torch.save(model, filename)
            self.last_checkpoint = datetime.datetime.now()
        if total_loss < self.best_loss:
            self.patience = 0
            self.best_loss = total_loss
            filename = f"{self.logdir}/{self.best_model_filename})"
            logger.debug(f"Saving best model (loss: {total_loss:02f}): {filename})")
            torch.save(model, filename)
        else:
            self.patience += 1
            if self.patience > self.max_patience:
                logger.debug(f"No patience stop here epoch {self.epoch})")
                return True
        return False

    def test(self):
        model = self.model.eval()
        total_loss = 0
        total_losses = defaultdict(int)
        total = 0
        for batch in tqdm.tqdm(self.test_loader, desc="test"):
            batch = self.to_device(model, batch)
            with torch.no_grad():
                loss, losses = self.get_loss(model, batch, reduction="mean")
                for _name, _loss in losses.items():
                    total_losses[_name] += _loss

            total_loss += loss
            total += 1

            if datetime.datetime.now() - self.last_train > self.train_interval:
                self.end_test(total_loss, total, total_losses, model)
                self.last_test = datetime.datetime.now()
                return
        self.end_test(total_loss, total, total_losses, model)
        self.last_test = datetime.datetime.now()

    def train_step(self, batch):
        model = self.model
        batch = self.to_device(model, batch)
        loss, losses = self.get_loss(model, batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if hasattr(self, "scheduler"):
            self.scheduler.step()

        self.writer.add_scalar("train/loss", loss, global_step=self.global_step)
        for loss_name, value in losses.items():
            self.writer.add_scalar(
                f"train/{loss_name}", value, global_step=self.global_step
            )

        self.global_step += self.batch_size

    def train(self):
        raise NotImplementedError(
            "Trainer does not know how to iterate over your dataloaders implement `train` method or use `EpochTrainer` or `TimedTrainer`"
        )


class EpochTrainer(Trainer):
    EPOCHS_DEFAULT = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_configuration()

    @classmethod
    def parser(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                description="EpochTrainer default arguments."
            )
        parser = super(EpochTrainer, cls).parser(parser)
        parser.add_argument(
            "--epochs",
            type=int,
            default=cls.EPOCHS_DEFAULT,
            help=f"(default {cls.EPOCHS_DEFAULT})",
        )
        return parser

    def split_dataset(self, dataset):
        N = len(dataset)
        n = int(self.train_proportion * N)

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n, N - n])
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader

    def test(self):
        model = self.model.eval()
        total_loss = 0
        total_losses = defaultdict(int)
        total = 0
        for batch in tqdm.tqdm(self.test_loader, desc="test"):
            batch = self.to_device(model, batch)
            with torch.no_grad():
                loss, losses = self.get_loss(model, batch, reduction="mean")
                for _name, _loss in losses.items():
                    total_losses[_name] += _loss

            total_loss += loss
            total += 1

        return self.end_test(total_loss, total, total_losses, model)

    def train(self):
        for epoch in tqdm.tqdm(range(self.epochs), desc="Train"):
            self.epoch = epoch
            self.model = self.model.train()
            for batch in tqdm.tqdm(self.train_loader):
                self.train_step(batch)

            early_stop = self.test()
            if early_stop:
                return


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, offset, offset_stop):
        self.dataset = dataset
        self.offset = offset
        self.offset_stop = offset_stop
        self.length = self.offset_stop - self.offset

    def __getitem__(self, idx):
        return self.dataset[idx + self.offset]

    def __len__(self):
        return self.length


class TimedTrainer(Trainer):
    TRAIN_TIME_PROPORTION_DEFAULT = 0.8
    TRAIN_INTERVAL_DEFAULT = datetime.timedelta(minutes=30)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_intervals()
        self.log_configuration()

    def setup_intervals(self):
        self.start_train = datetime.datetime.now()
        self.start_test = datetime.datetime.now()

        self.test_interval = (
            (1 - self.train_time_proportion)
            / self.train_time_proportion
            * self.train_interval
        )

    @classmethod
    def parser(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                description="EpochTrainer default arguments."
            )
        parser = super(TimedTrainer, cls).parser(parser)
        parser.add_argument(
            "--train-time-proportion",
            type=float,
            default=cls.TRAIN_TIME_PROPORTION_DEFAULT,
            help=f"Interval in minutes (default {cls.TRAIN_TIME_PROPORTION_DEFAULT})",
        )
        parser.add_argument(
            "--train-interval",
            type=datetime.timedelta,
            default=cls.TRAIN_INTERVAL_DEFAULT,
            help=f"Interval in minutes (default {cls.TRAIN_INTERVAL_DEFAULT})",
        )
        return parser

    def split_dataset(self, dataset):
        N = len(dataset)
        n = int(self.train_proportion * N)

        train_dataset = Subset(dataset, 0, n)
        test_dataset = Subset(dataset, n, N)

        train_sampler = RandomSampler(
            train_dataset, replacement=True, num_samples=int(1e7)
        )
        test_sampler = RandomSampler(
            test_dataset, replacement=True, num_samples=int(1e6)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=test_sampler,
        )
        return train_loader, test_loader

    def train(self):
        start_train = datetime.datetime.now()
        self.model = self.model.train()
        with tqdm.tqdm(
            desc="Train", total=int(self.train_interval.total_seconds())
        ) as progress:
            for batch in self.train_loader:
                self.train_step(batch)

                diff = datetime.datetime.now() - start_train
                if diff > self.train_interval:
                    break
                progress.update(diff.total_seconds() - progress.n)

        self.test()

    def test(self):
        model = self.model.eval()
        start_test = datetime.datetime.now()
        total_loss = 0
        total_losses = defaultdict(int)
        total = 0
        with tqdm.tqdm(
            desc="Test", total=int(self.test_interval.total_seconds())
        ) as progress:
            for batch in self.test_loader:
                batch = self.to_device(model, batch)
                with torch.no_grad():
                    loss, losses = self.get_loss(model, batch, reduction="mean")
                    for _name, _loss in losses.items():
                        total_losses[_name] += _loss

                total_loss += loss
                total += 1
                diff = datetime.datetime.now() - start_test
                if diff > self.test_interval:
                    break
                progress.update(diff.total_seconds() - progress.n)

        is_stop = self.end_test(total_loss, total, total_losses, model)
        if not is_stop:
            self.train()
