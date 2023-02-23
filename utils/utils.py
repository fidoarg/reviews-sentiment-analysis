import os
import yaml


def validate_config(config):
    """
    Takes as input the experiment configuration as a dict and checks for
    minimum acceptance requirements.
    Parameters
    ----------
    config : dict
        Experiment settings as a Python dict.
    """

    if "data" not in config:
        raise ValueError("Missing experiment data")

def load_config(config_file_path):
    """
    Loads experiment settings from a YAML file into a Python dict.
    See: https://pyyaml.org/.
    Parameters
    ----------
    config_file_path : str
        Full path to experiment configuration file.
        E.g: `/home/app/src/experiments/exp_001/config.yml`
    Returns
    -------
    config : dict
        Experiment settings as a Python dict.
    """
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"No file found with path: {config_file_path}")

    with open(file=config_file_path, mode='r') as yaml_file:
        yaml_data = yaml_file.read()
        config = yaml.safe_load(yaml_data) or {}
        
        # replace all 'none' strings with None value
        def replace_none(data):
            if isinstance(data, dict):
                return {k: replace_none(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [replace_none(item) for item in data]
            elif data == 'none':
                return None
            else:
                return data
        
        config = replace_none(config)

    # Don't remove this as will help you doing some basic checks on config
    # content
    validate_config(config)

    return config

def walkdir(folder):
    """
    Walk through all the files in a directory and its subfolders.
    Parameters
    ----------
    folder : str
        Path to the folder you want to walk.
    Returns
    -------
        For each file found, yields a tuple having the path to the file
        and the file name.
    """
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield (dirpath, filename)


def print_directory_structure(root_path, padding=''):
    # Get the list of files and directories in the root path
    files = os.listdir(root_path)

    # Loop through each file/directory and print its name
    for file in files:
        # Ignore hidden files/directories
        if file.startswith('.'):
            continue

        # Print the file/directory name
        print(padding + '|-- ' + file)

        # If the current item is a directory, recursively call this function to print its contents
        path = os.path.join(root_path, file)
        if os.path.isdir(path) and not 'data' in path and not 'venv' in path:
            print_directory_structure(path, padding + '    ')
