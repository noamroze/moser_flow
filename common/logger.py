
import os
import torch
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import matplotlib.pyplot as plt

class Logger:
    """Generic class to interface with various logging modules, e.g. wandb,
    tensorboard, etc.
    """

    def __init__(self, config):
        self.config = config

    def watch(self, model):
        """
        Monitor parameters and gradients.
        """
        raise NotImplementedError

    def log(self, update_dict, step=None, split=""):
        """
        Log some values.
        """
        assert step is not None
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict["{}/{}".format(split, key)] = update_dict[key]
            update_dict = new_dict
        return update_dict

    def log_plots(self, plots):
        raise NotImplementedError

    def close(self):
        pass

class TensorboardLogger(Logger):
    def __init__(self, config):
        super().__init__(config)
        self.writer = SummaryWriter(self.config["cmd"]["logs_dir"])

    # TODO: add a model hook for watching gradients.
    def watch(self, model):
        print("NOTE: model gradient logging to tensorboard not yet supported.")
        return False

    def log(self, update_dict, step=None, split=""):
        update_dict = super().log(update_dict, step, split)
        for key in update_dict:
            if torch.is_tensor(update_dict[key]):
                self.writer.add_scalar(key, update_dict[key].item(), step)
            elif isinstance(update_dict[key], dict):
                self.writer.add_scalars(key, update_dict[key], step)
            else:
                assert isinstance(update_dict[key], int) or isinstance(
                    update_dict[key], float
                )
                self.writer.add_scalar(key, update_dict[key], step)

    def log_figure(self, tag, fig, global_step):
        self.writer.add_figure(tag, fig, global_step)

class FileLogger(Logger):
    def __init__(self, config):
        super(FileLogger, self).__init__(config)
        self.log_dir = os.path.join(config["cmd"]["logs_dir"])
        self.log_path = os.path.join(self.log_dir, "training.log")
        self.train_scalars = defaultdict(list)
        self.val_scalars = defaultdict(list)

    def watch(self, model):
        print("NOTE: model gradient logging to file not yet supported.")
        return False

    def format(self, log_dict):
        log_str = []
        for k, v in log_dict.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    log_str.append("{}: {:.4f}".format("%s_%s" %(k, k2), v2))
            else:
                log_str.append("{}: {:.4f}".format(k, v))
        return ", ".join(log_str)

    def write(self, log_str):
        with open(self.log_path, "a") as f:
            f.write(log_str + "\n")

    def log(self, update_dict, step=None, split=""):
        if split=="train":
            self.write(self.format(update_dict))
            for key, value in update_dict.items():
                self.train_scalars[key].append(float(value))

        if split=="val":
            update_dict["step"] = step
            self.write("validation " + self.format(update_dict))
            for key, value in update_dict.items():
                self.val_scalars[key].append(float(value))
            
    def log_figure(self, *args, **kwargs):
        pass

    def close(self):
        for key, values in self.train_scalars.items():
            fig, ax = plt.subplots()
            ax.set_title(key)
            if isinstance(values[0], dict):
                for sub_key in values[0].items():
                    sub_values = [d[sub_key] for d in values]
                    ax.plot(values, title=sub_key)
            else:
                ax.plot(values)
            fig.savefig(os.path.join(self.log_dir, "training_%s.png" %key))

        for key, values in self.val_scalars.items():
            fig, ax = plt.subplots()
            ax.set_title(key)
            if isinstance(values[0], dict):
                for sub_key in values[0].keys():
                    sub_values = [d[sub_key] for d in values]
                    ax.plot(sub_values, label=sub_key)
                    ax.legend()
            else:
                ax.plot(values)
            fig.savefig(os.path.join(self.log_dir, "validation_%s.png" %key))
            
        plt.close("all")

        with open(os.path.join(self.log_dir, "train_scalars.pkl"), "wb") as f:
            pickle.dump(self.train_scalars, f)
        with open(os.path.join(self.log_dir, "val_scalars.pkl"), "wb") as f:
            pickle.dump(self.val_scalars, f)
        fig, ax = plt.subplots()
        ax.plot(self.train_scalars["total_time"], self.train_scalars["nll"])    
        ax.set_title("nll vs time")
        fig.savefig(os.path.join(self.log_dir, "train_nll_vs_time.png"))

        fig, ax = plt.subplots()
        cumulative_time = np.cumsum(self.train_scalars["training_step_time"])
        ax.plot(cumulative_time, self.train_scalars["nll"])    
        ax.set_title("nll vs time")
        fig.savefig(os.path.join(self.log_dir, "train_nll_vs_cumulative_time.png"))

        fig, ax = plt.subplots()
        ax.plot(self.val_scalars["total_time"], self.val_scalars["nll"])    
        ax.set_title("nll vs time")
        fig.savefig(os.path.join(self.log_dir, "val_nll_vs_time.png"))

        fig, ax = plt.subplots()
        ax.plot(self.val_scalars["cummulative_training_time"], self.val_scalars["nll"])    
        ax.set_title("nll vs training time")
        fig.savefig(os.path.join(self.log_dir, "val_nll_vs_cumulative_training_time.png"))
        
        plt.close('all')
