import argparse
import toml
from monai.config import print_config
from monai.utils import set_determinism

from utility import *

def inference():
    # get config file path from argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config', help="Path of config file.", required=True)
    args = parser.parse_args()
    config_file_path = args.config
    assert config_file_path.endswith(".toml"), "error: illegal config file path. (from inference.py)"
    
    # import configs from config file
    config = toml.load(config_file_path)
    globalVal.device = config['Inference']['device']

    # working paths
    task_name = config['Task']['task_name']
    globalVal.project_path = config['Task']['project_path']
    maybe_mkdir_p(globalVal.project_path)
    globalVal.model_path = join(globalVal.project_path, 'trained_models')
    maybe_mkdir_p(globalVal.model_path)
    globalVal.output_path = join(globalVal.project_path, 'inference', config['Inference']['inference_name'])
    maybe_mkdir_p(globalVal.output_path)
    # init log file, txt from print() will be automaticaly writhed into the log file
    init_log_file(globalVal.output_path, prefix='inference_log')

    # inference
    inference_name = config['Inference']['inference']
    inference_class = recursive_find_class(['inference'], inference_name, 'inference')
    assert inference_class is not None, "error: inference class not found. (from run_inference.py)"
    inference = inference_class(config)
    inference.inference()

if __name__ == "__main__":
    
    inference()
