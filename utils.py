import json
import os

def check_folder(path: str):
    """Method to create directory if it doesn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_opts(opts):
    path = os.path.join(opts.out, 'opts.json')
    with open(path, 'w') as f:
        json.dump(opts.__dict__, f, indent=2)
        print("Dumped command line arguments to {}".format(path))