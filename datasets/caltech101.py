import os
import pickle
import json
from collections import defaultdict
import random

from .utils import *
from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


class Caltech101(DatasetBase):
    dataset_dir = "caltech-101"

    def __init__(self, root, shot, seed, labels_list=None, model_name=None, subsample="all"):
        # Create model_suffix
        model_suffix = f"_{model_name}" if model_name else ""

        # Determine the number of hierarchy layers
        n_layers = len(labels_list) if labels_list is not None else 0

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")

        # Add model_suffix to file paths
        self.split_path = os.path.join(self.dataset_dir, f"split_Caltech101_{n_layers}layers{model_suffix}.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot{n_layers}layers{model_suffix}")
        mkdir_if_missing(self.split_fewshot_dir)

        # Get all class names
        categories = listdir_nohidden(self.image_dir)
        categories = [c for c in categories if c not in IGNORED]
        classnames = [NEW_CNAMES.get(c, c) for c in categories]
        self._classnames = classnames

        # Build multi-level classification hierarchy
        self.hierarchy = {}  # store hierarchical relationships (index-based)
        self.hierarchy_names = {}  # store superclass names at each level
        self.subclass_to_superclass_tensor = []  # store index mapping for each level

        if labels_list is not None:
            # Process the first level superclass-subclass relationship
            with open(labels_list[0], 'r') as f:
                first_level = json.load(f)

            # Build indices for first-level superclasses (following JSON order)
            superclass_names = list(first_level.keys())
            superclass_to_idx = {name: idx for idx, name in enumerate(superclass_names)}

            # Build mapping from subclass to first-level superclass
            subclass_to_superclass_idx = [-1] * len(classnames)
            # Store subclasses belonging to each superclass
            superclass_to_subclasses = {idx: [] for idx in range(len(superclass_names))}

            for superclass, subclasses in first_level.items():
                super_idx = superclass_to_idx[superclass]
                for subclass in subclasses:
                    # Find subclass index in classnames
                    for i, name in enumerate(classnames):
                        if name == subclass:
                            subclass_to_superclass_idx[i] = super_idx
                            superclass_to_subclasses[super_idx].append(name)
                            break

            # Store first level mapping
            self.subclass_to_superclass_tensor.append(subclass_to_superclass_idx)
            self.hierarchy[0] = superclass_names  # store first-level superclass names

            # Construct names for first-level superclasses (using subclass names)
            superclass_names_dict = {}
            for idx, subclasses in superclass_to_subclasses.items():
                if subclasses:  # ensure subclasses exist
                    combined_name = " or ".join(subclasses)
                    superclass_names_dict[idx] = combined_name
                else:
                    superclass_names_dict[idx] = f"layer1_{idx}"  # default name
            self.hierarchy_names[0] = superclass_names_dict

            # Process higher-level superclass relationships
            prev_superclass_to_subclasses = superclass_to_subclasses
            for level in range(1, n_layers):
                with open(labels_list[level], 'r') as f:
                    current_level = json.load(f)

                # Build indices for current level superclasses
                current_superclass_names = list(current_level.keys())
                current_superclass_to_idx = {name: idx for idx, name in enumerate(current_superclass_names)}

                # Build mapping from previous level superclasses to current level
                prev_to_current_idx = [-1] * len(superclass_names)
                # Store all base subclasses contained in each current superclass
                current_superclass_to_subclasses = {idx: [] for idx in range(len(current_superclass_names))}

                for superclass, subclasses in current_level.items():
                    super_idx = current_superclass_to_idx[superclass]
                    for subclass in subclasses:
                        if subclass in superclass_names:
                            prev_idx = superclass_names.index(subclass)
                            prev_to_current_idx[prev_idx] = super_idx
                            # Add all subclasses from previous superclass
                            if prev_idx in prev_superclass_to_subclasses:
                                current_superclass_to_subclasses[super_idx].extend(
                                    prev_superclass_to_subclasses[prev_idx]
                                )

                # Build subclass to current-level superclass mapping
                subclass_to_current_idx = [-1] * len(classnames)
                for i in range(len(classnames)):
                    prev_level_idx = self.subclass_to_superclass_tensor[level - 1][i]
                    if prev_level_idx != -1 and prev_level_idx < len(prev_to_current_idx):
                        subclass_to_current_idx[i] = prev_to_current_idx[prev_level_idx]

                # Store current level mapping
                self.subclass_to_superclass_tensor.append(subclass_to_current_idx)
                self.hierarchy[level] = current_superclass_names

                # Construct names for current-level superclasses
                # using "or" to concatenate all base subclass names
                current_names_dict = {}
                for idx, subclasses in current_superclass_to_subclasses.items():
                    if subclasses:
                        # remove duplicates since subclasses may repeat
                        unique_subclasses = list(set(subclasses))
                        combined_name = " or ".join(unique_subclasses)
                        current_names_dict[idx] = combined_name
                    else:
                        current_names_dict[idx] = f"layer{level + 1}_{idx}"  # default name
                self.hierarchy_names[level] = current_names_dict

                # Update variables for the next iteration
                superclass_names = current_superclass_names
                prev_superclass_to_subclasses = current_superclass_to_subclasses

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_data()
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = shot
        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    @property
    def classnames(self):
        return self._classnames

    def read_data(self):
        categories = listdir_nohidden(self.image_dir)
        categories = [c for c in categories if c not in IGNORED]
        categories.sort()  # ensure consistent class order

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                # Build multi-level label tuple
                subclass_idx = y  # subclass index (already 0-based)
                label_tuple = [subclass_idx]

                # Add superclass index for each hierarchy level
                for level_tensor in self.subclass_to_superclass_tensor:
                    if 0 <= subclass_idx < len(level_tensor):
                        label_tuple.append(level_tensor[subclass_idx])
                    else:
                        label_tuple.append(-1)

                label = tuple(label_tuple)
                item = Datum(impath=im, label=label, classname=c)
                items.append(item)
            return items

        train, val, test = [], [], []
        for y, category in enumerate(categories):
            category_dir = os.path.join(self.image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val

            # Ensure each split has at least one sample
            if n_train <= 0 or n_val <= 0 or n_test <= 0:
                print(f"Warning: Category {category} has insufficient images ({n_total}).")
                if n_total < 3:
                    # If too few images, place all in training set
                    n_train = n_total
                    n_val = 0
                    n_test = 0
                else:
                    # Ensure each split contains at least one sample
                    n_train = max(1, n_train)
                    n_val = max(1, n_val)
                    n_test = n_total - n_train - n_val

            # Use NEW_CNAMES mapping for class names
            class_name = NEW_CNAMES.get(category, category)

            train.extend(_collate(images[:n_train], y, class_name))
            if n_val > 0:
                val.extend(_collate(images[n_train:n_train + n_val], y, class_name))
            if n_test > 0:
                test.extend(_collate(images[n_train + n_val:], y, class_name))

        return train, val, test