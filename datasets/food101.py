import os
import pickle
import json
from collections import defaultdict
import random
import torch

from datasets.utils import *
from .oxford_pets import OxfordPets


class Food101(DatasetBase):
    dataset_dir = "food-101"

    def __init__(self, root, shot, seed, labels_list=None, label_path=None, model_name=None, subsample="all"):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        # Create model_suffix
        model_suffix = f"_{model_name}" if model_name else ""

        # Determine the number of layers and related paths
        if labels_list is not None:
            n_layers = len(labels_list)
            self.split_path = os.path.join(self.dataset_dir, f"split_zhou_Food101_{n_layers}layers{model_suffix}.json")
            self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot{n_layers}layers{model_suffix}")
        elif label_path is not None:
            n_layers = 1
            self.split_path = os.path.join(self.dataset_dir, f"split_zhou_Food101_1layer{model_suffix}.json")
            self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot1layer{model_suffix}")
        else:
            n_layers = 0
            self.split_path = os.path.join(self.dataset_dir, f"split_zhou_Food101{model_suffix}.json")
            self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot{model_suffix}")

        mkdir_if_missing(self.split_fewshot_dir)

        # Get all subclass names
        categories = sorted(os.listdir(self.image_dir))
        categories = [c for c in categories if os.path.isdir(os.path.join(self.image_dir, c))]
        self._classnames = categories

        # Build multi-level classification hierarchy
        self.hierarchy = {}  # store multi-level classification hierarchy (index-based)
        self.hierarchy_names = {}  # store superclass names at each level
        self.subclass_to_superclass_tensor = []  # store index mapping for each level

        # Process multi-level labels
        if labels_list is not None:
            # Process the relationship between first-level superclasses and subclasses
            with open(labels_list[0], 'r') as f:
                first_level = json.load(f)

            # Build indices for first-level superclasses following the JSON order
            superclass_names = list(first_level.keys())
            superclass_to_idx = {name: idx for idx, name in enumerate(superclass_names)}

            # Build mapping from subclasses to first-level superclasses
            subclass_to_superclass_idx = [-1] * len(categories)
            # Store subclass names contained in each superclass
            superclass_to_subclasses = {idx: [] for idx in range(len(superclass_names))}

            for superclass, subclasses in first_level.items():
                super_idx = superclass_to_idx[superclass]
                for subclass in subclasses:
                    # Find the index of the subclass in categories
                    for i, name in enumerate(categories):
                        if name == subclass:
                            subclass_to_superclass_idx[i] = super_idx
                            superclass_to_subclasses[super_idx].append(name)
                            break

            # Store the first-level mapping
            self.subclass_to_superclass_tensor.append(subclass_to_superclass_idx)
            self.hierarchy[0] = superclass_names  # store first-level superclass name list

            # Build names for first-level superclasses using subclass names
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

                # Build indices for current-level superclasses
                current_superclass_names = list(current_level.keys())
                current_superclass_to_idx = {name: idx for idx, name in enumerate(current_superclass_names)}

                # Build mapping from previous-level superclasses to current-level superclasses
                prev_to_current_idx = [-1] * len(superclass_names)
                # Store all base subclasses contained in each current-level superclass
                current_superclass_to_subclasses = {idx: [] for idx in range(len(current_superclass_names))}

                for superclass, subclasses in current_level.items():
                    super_idx = current_superclass_to_idx[superclass]
                    for subclass in subclasses:
                        if subclass in superclass_names:
                            prev_idx = superclass_names.index(subclass)
                            prev_to_current_idx[prev_idx] = super_idx
                            # Add all subclasses contained in the previous-level superclass to the current-level superclass
                            if prev_idx in prev_superclass_to_subclasses:
                                current_superclass_to_subclasses[super_idx].extend(
                                    prev_superclass_to_subclasses[prev_idx]
                                )

                # Build mapping from subclasses to current-level superclasses
                subclass_to_current_idx = [-1] * len(categories)
                for i in range(len(categories)):
                    prev_level_idx = self.subclass_to_superclass_tensor[level - 1][i]
                    if prev_level_idx != -1 and prev_level_idx < len(prev_to_current_idx):
                        subclass_to_current_idx[i] = prev_to_current_idx[prev_level_idx]

                # Store the current-level mapping
                self.subclass_to_superclass_tensor.append(subclass_to_current_idx)
                self.hierarchy[level] = current_superclass_names

                # Build names for current-level superclasses using "or" to connect all base subclass names
                current_names_dict = {}
                for idx, subclasses in current_superclass_to_subclasses.items():
                    if subclasses:
                        # Remove duplicates because repeated subclasses may exist
                        unique_subclasses = list(set(subclasses))
                        combined_name = " or ".join(unique_subclasses)
                        current_names_dict[idx] = combined_name
                    else:
                        current_names_dict[idx] = f"layer{level + 1}_{idx}"  # default name
                self.hierarchy_names[level] = current_names_dict

                # Update variables for the next iteration
                superclass_names = current_superclass_names
                prev_superclass_to_subclasses = current_superclass_to_subclasses

        # Process single-level labels (backward compatibility)
        elif label_path is not None:
            with open(label_path, 'r') as f:
                food_families = json.load(f)

            # Extract all subclass names from JSON in reading order
            classnames = []
            for family, subclasses in food_families.items():
                classnames.extend(subclasses)

            # Ensure there are no duplicate subclass names
            classnames = list(dict.fromkeys(classnames))
            self.classnames_s = classnames
            self.cname2lab = {c: i for i, c in enumerate(classnames)}

            # Load the JSON file of superclass-subclass relationships
            self.dtd_families = food_families

            # Create mapping from superclass names to superclass indices
            self.family_names = list(self.dtd_families.keys())
            self.classnames_f = self.family_names  # explicitly define superclass class names
            self.family2lab = {family: i for i, family in enumerate(self.family_names)}

            # Create mapping from subclasses to superclasses
            self.cname2family = {}
            for family, subclasses in self.dtd_families.items():
                for subclass in subclasses:
                    self.cname2family[subclass] = family

            # Create the tensor mapping subclasses to superclass indices
            num_subclasses = len(classnames)
            subclass_to_superclass_tensor = [-1] * num_subclasses
            for subclass_idx, subclass_name in enumerate(classnames):
                if subclass_name in self.cname2family:
                    family_name = self.cname2family[subclass_name]
                    family_idx = self.family2lab[family_name]
                    subclass_to_superclass_tensor[subclass_idx] = family_idx

            self.subclass_to_superclass_tensor = [subclass_to_superclass_tensor]
            self.hierarchy[0] = self.family_names
            self.hierarchy_names[0] = {idx: name for idx, name in enumerate(self.family_names)}

        # No-label case
        else:
            self.dtd_families = None
            self.cname2family = None
            self.family2lab = None
            self.classnames_f = None
            self.subclass_to_superclass_tensor = []
            print("Warning: No food101 JSON file provided. Will use original class labels only.")

        # Load or create dataset split
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_data()
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # Process few-shot learning
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

    def read_data(self):
        categories = sorted(os.listdir(self.image_dir))
        categories = [c for c in categories if os.path.isdir(os.path.join(self.image_dir, c))]

        p_trn, p_val = 0.5, 0.2
        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                # Build multi-level labels
                subclass_idx = y  # subclass index

                if self.subclass_to_superclass_tensor:
                    # Multi-level labels
                    label_tuple = [subclass_idx]
                    # Add the superclass index for each level
                    for level_tensor in self.subclass_to_superclass_tensor:
                        if 0 <= subclass_idx < len(level_tensor):
                            label_tuple.append(level_tensor[subclass_idx])
                        else:
                            label_tuple.append(-1)
                    label = tuple(label_tuple)
                else:
                    # Single-level label
                    label = subclass_idx

                item = Datum(impath=im, label=label, classname=c)
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(self.image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train: n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val:], label, category))

        return train, val, test