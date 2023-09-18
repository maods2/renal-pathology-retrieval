from pipelines.triplet import train_triplet
from pipelines.auto_encoder import train_auto_encoder

def get_pipeline(argument):
    if argument == "triplet":
        return train_triplet
    
    if argument == "autoencoder":
        return train_auto_encoder

    else:
        raise Exception("Not found")