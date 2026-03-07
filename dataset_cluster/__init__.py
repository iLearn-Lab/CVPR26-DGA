from .food101 import Food101
from .dtd import DescribableTextures
from .oxford_pets import OxfordPets
from .oxford_flowers import OxfordFlowers
from .fgvc_aircraft import FGVCAircraft
from .utils import *
from .stanford_cars import StanfordCars
from .eurosat import EuroSAT
from .caltech101 import Caltech101
from .ucf101 import UCF101
from .resisc45 import Resisc45
from .sun397 import SUN397
from .imagenet import ImageNet
basic_template = 'This is a photo of {}.'
dataset_list = {
                "oxford_pets": OxfordPets,
                "oxford_flowers": OxfordFlowers,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "dtd": DescribableTextures,
                "stanford_cars": StanfordCars,
                "eurosat": EuroSAT,
                "caltech101": Caltech101,
		"ucf101": UCF101,
                "resisc45": Resisc45,
                "sun397": SUN397,
                "imagenet": ImageNet
}

def build_cluster_dataset(dataset, root, shot=16, seed=0):
    return dataset_list[dataset](root, shot, seed)
