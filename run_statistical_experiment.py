import os
import matplotlib.pyplot as plt
import numpy as np
from common.flags import flags
from common.utils import build_config
from main import run, get_run_dir
import sys
import re
import torch

class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()    

N_RUNS = 5
def evaluate_checkpoints(checkpoint_dir):
    checkpoints = os.listdir(checkpoint_dir)
    checkpoint_re = re.compile("checkpoint_(\d+\.?\d*).pt")
    epochs = [float(checkpoint_re.match(checkpoint).groups()[0]) for checkpoint in checkpoints if checkpoint_re.match(checkpoint) is not None]
    idx = np.argsort(epochs)
    checkpoints = np.array(checkpoints)[idx]
    epochs = np.array(epochs)[idx]
    # times = []
    nlls = []
    test_nlls = []
    for i, checkpoint in enumerate(checkpoints):
        data = torch.load(os.path.join(checkpoint_dir, checkpoint))
        val_metrics = data["val_metrics"]
        nlls.append(val_metrics["nll"])

    test_data = torch.load(os.path.join(checkpoint_dir, "checkpoint_test_val.pt"))
    test_nll = test_data["metrics"]["nll"]
    ode_nll = test_data["metrics"]["ode_nll"]
    return epochs, nlls, test_nll, ode_nll


def main():
    parser = flags.get_parser()
    args = parser.parse_args()
    if not args.config_yml and not args.checkpoint:
        raise ValueError("either config-yml or checkpoint needs to be given")
    
    if args.checkpoint:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(os.path.dirname(checkpoint_dir), CONFIGURATION_FILENAME)
        args.config_yml = config_path
        args.run_dir = os.path.join(os.path.dirname(checkpoint_dir) + "_continued", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    config = build_config(args)
    run_dir = get_run_dir(config)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # log_file = open(os.path.join(run_dir, "out.log"), "a")
    log_file = Logger(os.path.join(run_dir, "out.log"))
    sys.stdout = log_file
    sys.stderr = log_file

    for i in range(N_RUNS):
        config_i = config.copy()
        config_i["cmd"]["run_dir"] = os.path.join(run_dir, f"run_{i}")
        config_i["optim"]["save_every_eval"] = True
        config_i["cmd"]["seed"] = i
        config_i["cmd"]["save_every"] = 100
        config_i["cmd"]["only_save_checkpoints"] = True
        config_i["optim"]["calc_ode_density"] = True
        print("running run %s" %i)
        run(config_i)
    analyze_runs(run_dir)

def analyze_runs(run_dir):
    best_nlls = []
    ode_nlls = []
    fig, ax = plt.subplots()
    for i in range(N_RUNS):
        run_dir_i = os.path.join(run_dir, f"run_{i}")
        checkpoint_dir = os.path.join(run_dir_i, "checkpoints")
        epochs, nlls, test_nll, ode_nll = evaluate_checkpoints(checkpoint_dir)
        ax.plot(epochs, nlls)
        best_nlls.append(test_nll)
        ode_nlls.append(ode_nll)

    fig.savefig(os.path.join(run_dir, "runs_statistics.png")) 
    mean_nll = np.mean(best_nlls)
    std_nll = np.std(best_nlls)
    print("nll is %s±%s" %(mean_nll, std_nll))

    mean_nll = np.mean(ode_nlls)
    std_nll = np.std(ode_nlls)
    print("ode nll is %s±%s" %(mean_nll, std_nll))




if __name__ == "__main__":
    main()
