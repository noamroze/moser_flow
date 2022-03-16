import os
import random
import time
import datetime
from collections import defaultdict, namedtuple
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from common.logger import TensorboardLogger, FileLogger
from common.utils import (
    save_checkpoint,
    EarlyStopping
)
from datasets import get_data_handler
import moser
import ffjord

class BaseRunner:
    def __init__(self, config, run_dir):
        self.cpu = config["cmd"]["cpu"]
        if torch.cuda.is_available() and not self.cpu:
            self.device = config["cmd"]["local_rank"]
            torch.cuda.set_device(self.device)
        else:
            self.device = "cpu"
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available
        
        
        print("running on device %s" %self.device)
        self.config = config

        self.config["cmd"]["checkpoint_dir"] = str(run_dir / "checkpoints")
        self.config["cmd"]["results_dir"] = str(run_dir / "results_dir")
        self.config["cmd"]["logs_dir"] = str(run_dir / "logs_dir")
        if not self.config["cmd"]["is_debug"]:
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)
        
        # saves last epoch for cases of continuing a saved experiment
        self.last_epoch = 0

        self.load()

    
    def load(self):
        self.load_seed_from_config()
        self.load_dataset()
        self.load_model()
        self.load_logger()
        self.load_optimizer()
        self.load_early_stopping()

    def load_early_stopping(self):
        if "max_time" in self.config["early_stop"]:
            self.early_stopping = TimeEarlyStopper(**self.config["early_stop"])
        else:   
            self.early_stopping = EarlyStopping(**self.config["early_stop"])

    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["cmd"]["seed"]
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self):
        self.loggers = []
        if not self.config["cmd"]["is_debug"]:
            for logger_name in self.config.get("logger", []):
                if logger_name == "tensorboard":
                    logger_class = TensorboardLogger
                elif logger_name == "file":
                    logger_class = FileLogger
                else:
                    raise ValueError("illegal logger %s" %self.config["logger"])
                self.loggers.append(logger_class(self.config))
        for logger in self.loggers:
            logger.watch(self.model)

    def load_dataset(self):
        print("### Loading dataset: {}".format(self.config["dataset"]["name"]))

        data_handler = get_data_handler(self.config["dataset"], eps=self.config["model"].get("eps", 0))
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = data_handler.get_dataloaders(
            batch_size=int(self.config["optim"]["batch_size"]),
            eval_batch_size=int(self.config["optim"]["eval_batch_size"]),
            shuffle=self.config["optim"].get("shuffle", True)
        )

    def load_model(self):
        raise NotImplementedError

    def load_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["optim"]["lr_initial"],  
            weight_decay=self.config["optim"]["weight_decay"]
        )
        if self.config["optim"].get("lr_milestones"):
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.config["optim"]["lr_milestones"],
                gamma=self.config["optim"]["lr_gamma"]
            )
        else:
            self.scheduler = None

    def load_pretrained(self, checkpoint_path=None, ddp_to_dp=False):
        if checkpoint_path is None or os.path.isfile(checkpoint_path) is False:
            print(f"Checkpoint: {checkpoint_path} not found!")
            return False

        print("### Loading checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))

        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.early_stopping.load_state_dict(checkpoint.get("early_stopping", {}))
        # self.config = checkpoint["config"]
        try:
            self.last_epoch = int(np.floor(checkpoint["epoch"]))
        except TypeError:
            self.last_epoch = 0
        return True

    def log(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log(*args, **kwargs)

    def save(self, epoch, metrics, name=None, split="val"):
        if name is None and self.config["cmd"]["only_save_checkpoints"]:
            name = epoch

        if not self.config["cmd"]["is_debug"]:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": self.config,
                    "metrics": metrics,
                    "early_stopping": self.early_stopping.state_dict()
                },
                self.config["cmd"]["checkpoint_dir"],
                name="%s_%s" %(name, split)
            )

    def calculate_loss(self, x):
        raise NotImplementedError

    def calc_metrics(self, *args, **kwargs):
        return {}

    def start(self):
        self.train_loader.dataset.initial_plots(self.config["cmd"]["results_dir"], model=self.model)
        self.cummulative_training_time = 0.
        self.init_tik = time.perf_counter() 

    def train(self):
        eval_every = self.config["optim"].get("eval_every", -1)
        
        iters = 0
        for epoch in range(self.last_epoch, self.config["optim"]["max_epochs"]):
            # Print metrics, make plots.
            if self.val_loader is not None and eval_every != -1 and epoch % eval_every == 0:
                val_metrics = self.validate(
                    split="val",
                    epoch=epoch,
                )
                self.early_stopping(**val_metrics)
                
                if self.config["optim"].get("save_all"):
                    self.save(epoch + 1, val_metrics)

                if self.early_stopping.is_best:
                    self.save(epoch + 1, val_metrics, name="best")

            if self.early_stopping.early_stop:
                break
            
            self.model.train()
            average_losses = defaultdict(lambda: 0)
            for (i, (x, _)) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                tik = time.perf_counter()
                loss_dict = self.calculate_loss(x.to(self.device))
                loss = loss_dict["loss"].mean()
                loss.backward()
                self.optimizer.step()
                tok = time.perf_counter()
                self.cummulative_training_time += tok - tik
                log_dict = {"epoch": epoch + (i + 1) / len(self.train_loader), "training_step_time": tok-tik, "total_time": tok-self.init_tik, "cummulative_training_time": self.cummulative_training_time}
                for key, value in loss_dict.items():
                    log_dict[key] = value.mean().item()
                    average_losses[key] += value.sum().item()
                log_dict.update(self.calc_metrics())
                # Evaluate on val set every `eval_every` iterations.
                iters += 1

            average_losses = {key: value / len(self.train_loader.dataset) for key, value in average_losses.items()}
            
            if epoch % self.config["cmd"]["print_every"] == 0:
                log_str = [
                    "{}: {:.4f}".format(k, v) for k, v in log_dict.items()
                ]
                print(", ".join(log_str))
                self.log(
                    log_dict,
                    step=epoch * len(self.train_loader) + i + 1,
                    split="train",
                )

            if self.config["optim"].get("save_every") and epoch % self.config["optim"].get("save_every") == 0:
                train_metrics = log_dict.copy()
                train_metrics.update(average_losses)
                self.save(epoch + 1, train_metrics, split='train')
            
            if log_dict["total_time"] > self.config["early_stop"].get("max_time", np.inf):
                break

            if self.scheduler:
                self.scheduler.step()
            torch.cuda.empty_cache()
            
        self.validate(split='val', epoch=epoch)
        self.finalize()

    def validate(self, split="val", epoch=None):
        print("### Evaluating on {}.".format(split))

        self.model.eval()
        if (epoch is None and self.config["cmd"].get("plot_at_test")) or (not self.config["cmd"]["only_save_checkpoints"]):
            if epoch is None:
                eval_dir = self.config["cmd"]["results_dir"]
            else:
                eval_dir = os.path.join(self.config["cmd"]["results_dir"], "epoch %s" %epoch)
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            self.train_loader.dataset.evaluate_model(self.model, epoch, eval_dir)
        
            if split == "test":
                self.train_loader.dataset.test_model(self.model, self.config["cmd"]["results_dir"], self.config["optim"].get("calc_ode_density", False))

        loader = self.val_loader if split == "val" else self.test_loader

        numel = 0
        log_dict = defaultdict(lambda: 0.)
        for (i, (x, _)) in tqdm(
            enumerate(loader),
            total=len(loader)
        ):
            loss_dict = self.calculate_loss(x.to(self.device))
            for key, value in loss_dict.items():
                value = value.detach()
                loss_dict[key] = value
                log_dict[key] += value.sum().item()
            numel += x.shape[0]
            del loss_dict
            torch.cuda.empty_cache()
        
        log_str = []
        for k,v in log_dict.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    v2 /= numel
                    log_dict[k][k2] = v2
                    log_str.append("{}: {:.4f}".format("%s_%s" %(k, k2), v2))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            else:
                v /= numel
                log_dict[k] = v
                log_str.append("{}: {:.4f}".format(k, v))

        total_time = time.perf_counter() - self.init_tik
        log_dict["total_time"] = total_time
        log_dict["cummulative_training_time"] = self.cummulative_training_time
        log_str.append("{}: {:.4f}".format("total_time", total_time))
        print(", ".join(log_str))

        # epoch is None for test split
        if epoch is not None:
            self.log(
                log_dict,
                step=(epoch + 1) * len(self.train_loader),
                split=split,
            )
        
        
        return dict(log_dict)    

    def finalize(self):
        for logger in self.loggers:
            logger.close()
        # load best model
        if self.config["optim"].get("test_on_best_val"):
            checkpoint_path = os.path.join(self.config["cmd"]["checkpoint_dir"], "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                self.load_pretrained(checkpoint_path)
        
        test_metrics = self.validate(split='test')
        self.save(metrics=test_metrics, epoch=None, name="test")


class MoserEarlyStopper(EarlyStopping):
    def __call__(self, **loss_dict):
        loss_dict.pop("loss")
        super().__call__(loss_dict["nll"], **loss_dict)
    

class TimeEarlyStopper(EarlyStopping):
    def __init__(self, max_time, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_time = max_time

    def __call__(self, loss, total_time, **kwargs):
        super().__call__(loss, **kwargs)
        if total_time > self.max_time:
            self.early_stop = True


class MoserRunner(BaseRunner):
    def load_early_stopping(self):
        if "max_time" in self.config["early_stop"]:
            self.early_stopping = TimeEarlyStopper(**self.config["early_stop"])
        else:
            self.early_stopping = MoserEarlyStopper(**self.config["early_stop"])

    def load_model(self):
        kwargs = {}
        MODELS = {
            "torus": moser.TorusMoserFlow,
            "implicit": moser.ImplicitMoser
        }
        model_class = MODELS[self.config["model"]["manifold"]]
        if self.config["model"]["manifold"] == "implicit":
            surface = self.train_loader.dataset.surface
            surface.to(self.device)
            kwargs["surface"] = surface
        self.model = model_class(self.config["dataset"]["input_dim"],
                            self.config["model"], self.device, **kwargs)

    
    def calculate_loss(self, x):
        nll = self.model(x.to(self.device))
        mc_batch_size_scale = self.config["optim"].get("mc_batch_size_scale", 1)
        lambda_plus = self.config["optim"].get("lambda_plus", 0)
        lambda_minus = self.config["optim"].get("lambda_minus", 1)
        positivity_loss = self.model.positivity_loss(lambda_plus, lambda_minus, x.shape[0] * mc_batch_size_scale)
        loss =  nll.mean() + positivity_loss.mean()
    
        # Print metrics, make plots.
        loss_dict = {
            "loss": loss,
            "nll": nll,
            "positivity": positivity_loss,
        }
        return loss_dict

    def validate(self, split="val", epoch=None):
        val_metrics = super().validate(split, epoch)
        if split == "test" and self.config["optim"].get("calc_ode_density"):
            nll = 0.
            numel = 0
            for (i, (x, _)) in tqdm(
                enumerate(self.test_loader),
                total=len(self.test_loader)
            ):
                nll -= self.model.direct_log_likelihood(x.to(self.device)).sum()
                numel += x.shape[0]

            nll /= numel
            print("test nll by ode is %s, estimated nll is %s" % (nll, val_metrics["nll"]))
            val_metrics["ode_nll"] = nll.item()
        else:
            val_metrics["ode_nll"] = 0.
        return val_metrics

class FFJORDRunner(BaseRunner):
    def __init__(self, *largs, **kwargs):
        super(FFJORDRunner, self).__init__(*largs, **kwargs)

    def load_model(self):
        args_dict = self.config["model"]
        args = namedtuple("Args", args_dict.keys())(**args_dict)
        self.model = ffjord.FFJORDTorusModel(args, self.config["dataset"]["input_dim"], self.device)

    def calculate_loss(self, x):
        loss = self.model(x)
        return {"loss": loss, "nll": loss}

    def calc_metrics(self):
        return {"nfe": ffjord.count_nfe(self.model)}
