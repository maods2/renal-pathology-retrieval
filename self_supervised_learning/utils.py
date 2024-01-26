import sys

sys.path.append('../embeddings/')

from pathlib import Path
import torch
import time



def slice_image_paths(paths):
    return [i.split('/')[-1].replace('\\','/') for i in paths]

def create_timestamp_folder():
    """
    Create a folder name based on the current timestamp.
    Returns:
        folder_name (str): The name of the folder, in the format 'YYYY-MM-DD-HH-MM-SS'.
    """
    current_time = time.localtime()
    folder_name = time.strftime('%Y-%m-%d-%H-%M-%S', current_time)
    return f'efficient_SimCLR_{folder_name}'    

def save_checkpoint(model, optimizer):
    timestamp_folder = create_timestamp_folder()
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer is not None else None}
    Path(f'./data_output/checkpoints/{timestamp_folder}').mkdir(exist_ok=True)
    filename=f"./data_output/checkpoints/{timestamp_folder}/checkpoint.pth.tar"
    # draw_loss_curve(
    #     history=loss, 
    #     results_path=f"./data_output/checkpoints/{timestamp_folder}"
    #     )
    torch.save(state, filename)

    # with open(f"./data_output/checkpoints/{timestamp_folder}/config.yaml", "w") as yaml_file:
    #     yaml.dump(config.__dict__, yaml_file)

