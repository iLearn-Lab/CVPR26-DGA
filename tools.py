from datasets import build_dataset, build_data_loader
from dataset_cluster import build_cluster_dataset, build_cluster_loader
import torch
import random
import numpy as np
import clip
import json
import torch.nn as nn
from torch.nn import functional as F
import math
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, RandomResizedCrop, \
    RandomHorizontalFlip
from methods.vp import PaddingVR
from torchvision.transforms import InterpolationMode
from PIL import Image



class TextProjection(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.projection = nn.Linear(emb_dim, emb_dim, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(self, x):
        text = self.projection(x)
        return text

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def clip_classifier(classnames, template, clip_model):
    '''
    Text encoder for label-based classification

    params:
    classnames: class name of the label space
    template: text prompts
    clip_model: the pretrained CLIP
    '''

    device = next(clip_model.parameters()).device
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)

            # Calculate text embedding
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights

def clip_attr_classifier(classnames, clip_model, dir, num_attr):
    '''
    Text encoder for attribute-based classification.

    params:
    classnames: class name of the label space
    clip_model: the pretrained CLIP
    dir: attribute path
    num_attr: 'm' in the paper, the attribute number
    '''

    data = json.load(open(dir, 'r'))
    device = next(clip_model.parameters()).device
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize
            filtered_sentences = [sentence for sentence in data[classname] if len(sentence) >= 5]
            if len(filtered_sentences) == 0:
                print(classname)
                raise ValueError("No valid attributes have been generated")
            if num_attr < len(filtered_sentences):
                texts = random.sample(filtered_sentences, num_attr)
            else:
                texts = random.choices(filtered_sentences, k=num_attr)
            texts = clip.tokenize(texts).to(device)

            # Calculate text embedding
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def aggregate_subclass_embeddings_to_superclass(
        num,
        i,
        txt_emb_desattr,
        txt_emb_distattr,
        subclass_to_superclass):
    """
    Aggregate subclass text embeddings into superclass representations.

    All attribute embeddings of subclasses belonging to the same superclass
    are concatenated together.

    Args:
        txt_emb_desattr: Descriptive attribute embeddings of subclasses
                         shape (num_attr, num_subclasses, embed_dim)

        txt_emb_distattr: Discriminative attribute embeddings of subclasses
                          shape (num_attr, num_subclasses, embed_dim)

        subclass_to_superclass: Mapping from subclass index to superclass index

    Returns:
        superclass_txt_emb_desattr:
            Descriptive attribute embeddings for superclasses
            shape (total_superclass_attrs, num_superclasses, embed_dim)

        superclass_txt_emb_distattr:
            Discriminative attribute embeddings for superclasses
            shape (total_superclass_attrs, num_superclasses, embed_dim)
    """

    device = txt_emb_desattr.device

    subclass_to_superclass_mapping = subclass_to_superclass

    # Convert mapping to tensor
    if not isinstance(subclass_to_superclass_mapping, torch.Tensor):
        subclass_to_superclass_mapping = torch.tensor(
            subclass_to_superclass_mapping,
            device=device,
            dtype=torch.long
        )
    else:
        subclass_to_superclass_mapping = subclass_to_superclass_mapping.to(device)

    num_attr, num_subclasses, embed_dim = txt_emb_desattr.shape

    num_superclasses = subclass_to_superclass_mapping.max().item() + 1

    # Collect attribute features for each superclass
    superclass_desattr_list = []
    superclass_distattr_list = []

    for superclass_id in range(num_superclasses):

        subclass_mask = (subclass_to_superclass_mapping == superclass_id)
        subclass_indices = torch.where(subclass_mask)[0]

        # Extract all subclass attributes belonging to this superclass
        subclass_desattr = txt_emb_desattr[:, subclass_indices, :]   # (num_attr, num_subclasses_in_group, embed_dim)
        subclass_distattr = txt_emb_distattr[:, subclass_indices, :] # (num_attr, num_subclasses_in_group, embed_dim)

        # Flatten attributes across subclasses
        # (num_attr, num_subclasses, embed_dim) -> (num_attr * num_subclasses, 1, embed_dim)
        subclass_desattr_flattened = subclass_desattr.permute(1, 0, 2).reshape(
            -1, 1, embed_dim
        )

        subclass_distattr_flattened = subclass_distattr.permute(1, 0, 2).reshape(
            -1, 1, embed_dim
        )

        target_count = 2 ** (i + 1)
        current_count = len(subclass_indices)

        if current_count != target_count:

            # Target number of attribute embeddings
            target_size = num * target_count
            current_size = subclass_desattr_flattened.size(0)

            # Create repeated indices to reach the target size
            repeat_indices = torch.arange(
                current_size,
                device=subclass_desattr_flattened.device
            ).repeat(
                (target_size + current_size - 1) // current_size
            )[:target_size]

            subclass_desattr_flattened = subclass_desattr_flattened[repeat_indices]
            subclass_distattr_flattened = subclass_distattr_flattened[repeat_indices]

        superclass_desattr_list.append(subclass_desattr_flattened)
        superclass_distattr_list.append(subclass_distattr_flattened)

    superclass_txt_emb_desattr = torch.stack(superclass_desattr_list, dim=1).squeeze(dim=2).to(device)
    superclass_txt_emb_distattr = torch.stack(superclass_distattr_list, dim=1).squeeze(dim=2).to(device)

    return superclass_txt_emb_desattr, superclass_txt_emb_distattr

def getLogits(num_attr, exp, x_emb, t_emb, k=3):

    fea_dim = t_emb.shape[-1]
    t_emb = t_emb.permute(2, 1, 0).reshape(fea_dim, -1)
    res = exp * x_emb @ t_emb
    bs = res.shape[0]
    res = res.reshape(bs, -1, num_attr)
    res, _ = torch.sort(res, dim=-1, descending=True)
    logits = torch.mean(res[:, :, :k], dim=2)
    return logits


def build_loader(dataset_name, root_path, train_preprocess=None, test_preprocess=None, batch_size=64, shot=16, seed=0, labels_list=None, model_name=None):
    '''
    Retuen the loader of downstream tasks.

    params:
    dataset_name: downstream dataset name
    root_path: the path of dataset
    train_preprocess/test_preprocess: the data argumentation performed on samples
    batch_size: training batch size
    shot: the available number of samples per class
    seed: the random seed
    '''
    dataset = build_dataset(dataset_name, root_path, shot, seed, labels_list, model_name)
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=batch_size, is_train=True, tfm=train_preprocess, shuffle=True)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=batch_size, is_train=False, tfm=test_preprocess, shuffle=True)
    return train_loader, test_loader, dataset.classnames, dataset.hierarchy_names, dataset.subclass_to_superclass_tensor

def build_loader_cluster(dataset_name, root_path, batch_size=64, shot=16, seed=0, cluster_preprocess=None):
    dataset = build_cluster_dataset(dataset_name, root_path, shot, seed)
    cluster_loader = build_cluster_loader(data_source=dataset.test, batch_size=batch_size, is_train=False, tfm=cluster_preprocess, shuffle=False)
    return cluster_loader, dataset.classnames

