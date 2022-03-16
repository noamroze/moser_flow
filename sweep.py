import yaml
import os
import subprocess
from datetime import datetime
from common.utils import create_grid, build_user_config

VENV = "moser"


class ScriptJob(object):
    def __init__(self, name, script, config, params, run_dir):
        self.job_name = name
        self.script = script
        self.params = params
        self.run_dir = run_dir
        self.config = config
        

    def create_sh_file(self, commands=None):
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        config_path = os.path.join(self.run_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config, f)
        self.params['config-yml'] = config_path
        self.params['run-dir'] = self.run_dir
        
        if not commands:
            commands = []
        if isinstance(self.params, dict):
            script_params = " ".join(["--%s %s" %(key, value) for key,value in self.params.items()])
        elif isinstance(self.params, list):
            script_params = " ".join(self.params)
        else:
            raise Exception
        bash_commands = ['echo "=================================="'] + commands + [
            'conda activate %s' % VENV, 
            "python %s %s" % (self.script, script_params)
        ]

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        bash_file = os.path.join(self.run_dir, "run.sh")
        with open(bash_file, "w") as text_file:
            text_file.write('\n'.join(bash_commands))
        return bash_file

class VanillaJobRunner:
    def run_job(self, job):
        sh_path = job.create_sh_file()
        out_path = os.path.join(job.run_dir, "out.log")
        err_path = os.path.join(job.run_dir, "err.log")
        subprocess.Popen(["bash", sh_path], stdout=open(out_path, "a"), stderr=open(err_path, "a"))
        # os.system("bash %s" %sh_path)

def create_sweep_jobs(args, run_dir):
    base_config = build_user_config(args.config_yml, args.config_override)
    # base_config["identifier"] = identifier
    grid_configs = create_grid(base_config, args.sweep_yml)
    
    timestamp = datetime.fromtimestamp(datetime.now().timestamp()).strftime(
                "%Y-%m-%d-%H-%M-%S"
        )
    dir_name = "%s_%s" %(timestamp, args.identifier)
    run_dir = os.path.join(run_dir, dir_name)
    jobs = []
    for i, config in enumerate(grid_configs):
        identifier = args.identifier
        params = {
            "mode": "train",
            "identifier": identifier + "_run_%s" %i,
            "print-every": args.print_every
            }
        if args.only_save_checkpoints:
            params["only_save_checkpoints"] = ""
        job = ScriptJob(
            name="moser_%s" %args.identifier,
            script=os.path.join(os.getcwd(), "main.py"),
            config=config,
            params=params,
            run_dir=os.path.join(run_dir, "run_%s" %i),
        )
        jobs.append(job)
    return jobs
        