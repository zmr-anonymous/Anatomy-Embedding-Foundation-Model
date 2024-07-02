import argparse
import toml
from monai.config import print_config
from monai.utils import set_determinism

from utility import *

def run_train():
    # get config file path from argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config', help="Path of config file.", required=True)
    args = parser.parse_args()
    config_file_path = args.config
    assert config_file_path.endswith(".toml"), "error: illegal config file path. (from run_training.py)"
    
    # import configs from config file
    config = toml.load(config_file_path)
    globalVal.device = config['Train']['device']

    # working paths
    task_name = config['Task']['task_name']
    globalVal.project_path = config['Task']['project_path']
    maybe_mkdir_p(globalVal.project_path)
    globalVal.model_path = join(globalVal.project_path, 'trained_models', task_name)
    maybe_mkdir_p(globalVal.model_path)
    
    # init log file, txt from print() will be automaticaly writhed into the log file
    init_log_file(globalVal.model_path, prefix='training_log')

    # trainer
    trainer_name = config['Train']['trainer']
    trainer_class = recursive_find_class(['trainer'], trainer_name, 'trainer')
    assert trainer_class is not None, "error: trainer class not found. (from run_training.py)"
    trainer = trainer_class(config)

    # train
    trainer.run_training()


if __name__ == "__main__":
    
    run_train()
