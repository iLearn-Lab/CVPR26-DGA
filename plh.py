from torchvision.transforms import Compose, Resize, Lambda, ToTensor, FiveCrop, InterpolationMode, Normalize
import torch
from cfg import DOWNSTREAM_PATH
import random
import numpy as np
import os
import json
from tqdm import tqdm
from torch.nn import functional as F
from tools import build_loader_cluster
from torchvision import transforms
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Build prototype-based hierarchical labels with CLIP features.")

    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="Dataset name(s), e.g. --dataset fgvc or --dataset fgvc eurosat dtd"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/16",
        help="CLIP model name, e.g. ViT-B/16 or ViT-B/32"
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of hierarchy levels"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/home/user13/clip_model",
        help="Directory containing CLIP .pt model files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=16,
        help="Shot number for build_loader_cluster"
    )

    return parser.parse_args()


def set_seed(seed):
    """Set random seed"""
    print("Setting fixed seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name, model_path):
    """Load CLIP model"""
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if model_name == "ViT-B/16":
        modelname = "vit_b_16"
        model_file = "vit_b_16.pt"
        input_resolution = 224
    elif model_name == "ViT-B/32":
        modelname = "vit_b_32"
        model_file = "vit_b_32.pt"
        input_resolution = 224
    else:
        modelname = model_name
        model_file = f"{model_name}.pt"
        input_resolution = 224

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    def create_preprocess(input_resolution):
        return transforms.Compose([
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
            FiveCrop(input_resolution),
            Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
            Normalize(mean=mean, std=std),
        ])

    local_model_path = os.path.join(model_path, model_file)

    print(f"Loading model from local path: {local_model_path}")
    model_cluster = torch.jit.load(local_model_path, map_location=device)
    model_cluster.eval()
    model_cluster = model_cluster.to(device)

    preprocess = create_preprocess(input_resolution)

    return model_cluster, modelname, preprocess, device


def compute_distance_matrix(prototypes):
    """
    Compute pairwise Euclidean distance matrix between prototypes

    Input:
        prototypes (N, D) numpy array

    Output:
        distance matrix (N, N)
    """
    protos_exp1 = prototypes[:, np.newaxis]
    protos_exp2 = prototypes[np.newaxis, :]
    dist_matrix = np.sqrt(np.sum((protos_exp1 - protos_exp2) ** 2, axis=-1))
    return dist_matrix


def build_hierarchy_levels(prototypes, classnames, n_layers=3):
    """
    Build hierarchical clustering with pairwise merging

    Args:
        prototypes: (num_classes, feature_dim)
        classnames: list of class names
        n_layers: number of hierarchy levels

    Returns:
        all_levels
        expanded_levels
    """
    num_classes = len(classnames)
    all_levels = []
    expanded_levels = []

    # first level: start from original classes
    level1_groups = []
    remaining_indices = list(range(num_classes))
    dist_matrix = compute_distance_matrix(prototypes)

    # avoid self merge
    np.fill_diagonal(dist_matrix, np.inf)

    # strict pairwise merging
    while len(remaining_indices) > 1:
        min_i, min_j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        if min_i in remaining_indices and min_j in remaining_indices:
            level1_groups.append([min_i, min_j])

            remaining_indices.remove(min_i)
            remaining_indices.remove(min_j)

            dist_matrix[min_i, :] = np.inf
            dist_matrix[:, min_i] = np.inf
            dist_matrix[min_j, :] = np.inf
            dist_matrix[:, min_j] = np.inf
        else:
            dist_matrix[min_i, min_j] = np.inf
            dist_matrix[min_j, min_i] = np.inf

    if len(remaining_indices) == 1:
        level1_groups.append([remaining_indices[0]])

    all_levels.append(level1_groups)
    expanded_levels.append(level1_groups)

    # build higher levels
    for layer in range(1, n_layers):
        print(f"Building hierarchy level {layer + 1}...")

        prev_groups = all_levels[layer - 1]
        group_prototypes = []

        for group in prev_groups:
            if len(group) == 1:
                if layer == 1:
                    group_proto = prototypes[group[0]]
                else:
                    original_classes = []
                    for idx in group:
                        original_classes.extend(expanded_levels[0][idx])
                    group_proto = np.mean([prototypes[idx] for idx in original_classes], axis=0)
            else:
                if layer == 1:
                    group_proto = np.mean([prototypes[idx] for idx in group], axis=0)
                else:
                    original_classes = []
                    for idx in group:
                        original_classes.extend(expanded_levels[0][idx])
                    group_proto = np.mean([prototypes[idx] for idx in original_classes], axis=0)

            group_prototypes.append(group_proto)

        group_prototypes = np.array(group_prototypes)

        dist_matrix = compute_distance_matrix(group_prototypes)
        np.fill_diagonal(dist_matrix, np.inf)

        current_groups = []
        remaining_indices = list(range(len(prev_groups)))

        while len(remaining_indices) > 1:
            min_i, min_j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

            if min_i in remaining_indices and min_j in remaining_indices:
                current_groups.append([min_i, min_j])

                remaining_indices.remove(min_i)
                remaining_indices.remove(min_j)

                dist_matrix[min_i, :] = np.inf
                dist_matrix[:, min_i] = np.inf
                dist_matrix[min_j, :] = np.inf
                dist_matrix[:, min_j] = np.inf
            else:
                dist_matrix[min_i, min_j] = np.inf
                dist_matrix[min_j, min_i] = np.inf

        if len(remaining_indices) == 1:
            current_groups.append([remaining_indices[0]])

        all_levels.append(current_groups)

        expanded_current = []
        for group in current_groups:
            expanded_group = []
            for idx in group:
                expanded_group.extend(expanded_levels[layer - 1][idx])
            expanded_current.append(expanded_group)

        expanded_levels.append(expanded_current)

    return all_levels, expanded_levels


def save_hierarchy_to_files(all_levels, classnames, dataset_name, modelname, n_layers=3):
    """Save hierarchy results to JSON files"""
    save_dir = os.path.join("./labels_clip", modelname, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    for layer in range(n_layers):
        layer_dict = {}

        if layer == 0:
            for i, group in enumerate(all_levels[layer]):
                layer_dict[f"layer{layer + 1}_{i}"] = [classnames[idx] for idx in group]
        else:
            for i, group in enumerate(all_levels[layer]):
                layer_dict[f"layer{layer + 1}_{i}"] = [f"layer{layer}_{idx}" for idx in group]

        save_path = os.path.join(save_dir, f"layer{layer + 1}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(layer_dict, f, indent=2, ensure_ascii=False)

        print(f"Level {layer + 1} hierarchy saved to {save_path}")


def extract_features_for_dataset(dataset_name, model_cluster, device, batch_size=64, shot=16, seed=0):
    """Extract CLIP features for a dataset"""
    print(f"\nProcessing dataset: {dataset_name}")

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    cluster_process = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        FiveCrop(224),
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
        Normalize(mean=mean, std=std),
    ])

    try:
        cluster_loader, classnames = build_loader_cluster(
            dataset_name,
            DOWNSTREAM_PATH,
            batch_size=batch_size,
            shot=shot,
            seed=seed,
            cluster_preprocess=cluster_process
        )
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        return None, None, None

    features = []
    sample_indices = []

    print(f"Start extracting features for {dataset_name}...")

    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(tqdm(cluster_loader, desc=f"Extracting {dataset_name} features")):
            imgs = imgs.to(device)

            if len(imgs.shape) == 5:
                B, N = imgs.shape[:2]
                imgs = imgs.view(-1, *imgs.shape[2:])
            else:
                B = imgs.shape[0]
                N = 1

            image_features = model_cluster.encode_image(imgs)

            if N > 1:
                image_features = image_features.view(B, N, -1)
                feats_mean = image_features.mean(dim=1)
            else:
                feats_mean = image_features

            feats_mean = F.normalize(feats_mean, dim=1)

            features.append(feats_mean.cpu())

            if hasattr(target, "cpu"):
                sample_indices.extend(target.cpu().numpy().tolist())
            else:
                sample_indices.extend(target)

    features = torch.cat(features, dim=0).numpy()
    sample_indices = np.array(sample_indices)

    num_classes = len(classnames)
    print(f"{dataset_name} feature extraction finished: {len(features)} samples, {num_classes} classes")

    return features, sample_indices, classnames


def compute_class_prototypes(features, sample_indices, classnames):
    """Compute prototype for each class"""
    num_classes = len(classnames)
    class_prototypes = []

    for i in range(num_classes):
        class_feats = features[sample_indices == i]

        if len(class_feats) == 0:
            print(f"Warning: class {i} ({classnames[i]}) has no samples!")
            class_prototypes.append(np.zeros(features.shape[1]))
        else:
            class_prototypes.append(class_feats.mean(axis=0))

    return np.stack(class_prototypes, axis=0)


def process_single_dataset(dataset_name, model_cluster, modelname, device, n_layers, batch_size=64, shot=16, seed=0):
    """Process one dataset"""
    try:
        features, sample_indices, classnames = extract_features_for_dataset(
            dataset_name=dataset_name,
            model_cluster=model_cluster,
            device=device,
            batch_size=batch_size,
            shot=shot,
            seed=seed,
        )

        if features is None:
            print(f"Skipping dataset {dataset_name}")
            return False

        class_prototypes = compute_class_prototypes(features, sample_indices, classnames)

        print(f"Building hierarchical clustering for {dataset_name}...")
        all_levels, expanded_levels = build_hierarchy_levels(
            class_prototypes,
            classnames,
            n_layers=n_layers
        )

        save_hierarchy_to_files(
            all_levels,
            classnames,
            dataset_name,
            modelname,
            n_layers=n_layers
        )

        print(f"{dataset_name} finished!")
        return True

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        return False


def main():
    """Main function"""
    args = parse_args()

    dataset_list = args.dataset
    seed = args.seed
    model_name = args.model_name
    n_layers = args.n_layers
    model_path = args.model_path
    batch_size = args.batch_size
    shot = args.shot

    set_seed(seed)

    print("Loading CLIP model...")
    model_cluster, modelname, _, device = load_model(model_name, model_path)
    print("Model loaded")

    total_datasets = len(dataset_list)
    successful_datasets = []
    failed_datasets = []

    print(f"\nStart processing {total_datasets} datasets...")
    print("=" * 60)

    for i, dataset_name in enumerate(dataset_list, 1):
        print(f"\n[{i}/{total_datasets}] Processing: {dataset_name}")
        print("-" * 40)

        success = process_single_dataset(
            dataset_name=dataset_name,
            model_cluster=model_cluster,
            modelname=modelname,
            device=device,
            n_layers=n_layers,
            batch_size=batch_size,
            shot=shot,
            seed=seed,
        )

        if success:
            successful_datasets.append(dataset_name)
        else:
            failed_datasets.append(dataset_name)

        print("-" * 40)

    print("\n" + "=" * 60)
    print("Processing finished!")
    print(f"Successfully processed: {len(successful_datasets)}/{total_datasets}")

    if successful_datasets:
        print("\nSuccessful datasets:")
        for dataset in successful_datasets:
            print(f"  - {dataset}")

    if failed_datasets:
        print("\nFailed datasets:")
        for dataset in failed_datasets:
            print(f"  - {dataset}")

    print(f"\nAll label files saved to ./labels_clip/{modelname}/")


if __name__ == "__main__":
    main()