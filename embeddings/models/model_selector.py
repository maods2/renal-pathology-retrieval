
from models.vgg16 import get_vgg16
from models.efficient_netb0 import get_efficient_netb0
from models.vit import get_vit

def get_model(argument):
    if argument == "vgg16":
        return get_vgg16()

    elif argument == "efficientnetb0":
        return get_efficient_netb0()

    elif argument == "vit":
        return get_vit()
        
    else:
        raise Exception("Not found")