import os
import pickle
import json
import torch
from scipy.io import loadmat
from collections import defaultdict

from datasets.utils import *
from .oxford_pets import OxfordPets


class StanfordCars(DatasetBase):
    dataset_dir = "stanford_cars"

    def __init__(self, root, shot, seed, labels_list=None, model_name=None, subsample="all"):
        n_layers = len(labels_list) if labels_list else 1
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # Add model_name to the split file path
        model_suffix = f"_{model_name}" if model_name else ""
        self.split_path = os.path.join(self.dataset_dir, f"split_StanfordCars_{n_layers}layers{model_suffix}.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot{n_layers}layers{model_suffix}")

        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        mkdir_if_missing(self.split_fewshot_dir)

        # Initialize a list to store subclass folder names
        classnames = []

        # Traverse all subfolders under the specified directory
        for folder_name in sorted(os.listdir(self.train_dir)):
            folder_path = os.path.join(self.train_dir, folder_name)
            # Check whether it is a directory rather than a file
            if os.path.isdir(folder_path):
                classnames.append(folder_name)

        self._classnames = classnames
        self.cname2lab = {cname: lab for lab, cname in enumerate(classnames)}

        # Build multi-level hierarchy relationships
        self.hierarchy = {}  # store multi-level hierarchy relationships (index-based)
        self.hierarchy_names = {}  # store superclass names at each level
        self.subclass_to_superclass_tensor = []  # store index mappings for each level

        if labels_list is not None:
            # Process the relationship between first-level superclasses and subclasses
            with open(labels_list[0], 'r') as f:
                first_level = json.load(f)

            # Build indices for first-level superclasses following the JSON order
            superclass_names = list(first_level.keys())
            superclass_to_idx = {name: idx for idx, name in enumerate(superclass_names)}

            # Build mapping from subclasses to first-level superclasses
            subclass_to_superclass_idx = [-1] * len(classnames)
            # Store subclass names contained in each superclass
            superclass_to_subclasses = {idx: [] for idx in range(len(superclass_names))}

            for superclass, subclasses in first_level.items():
                super_idx = superclass_to_idx[superclass]
                for subclass in subclasses:
                    # Find the index of the subclass in classnames
                    if subclass in self.cname2lab:
                        subclass_idx = self.cname2lab[subclass]
                        subclass_to_superclass_idx[subclass_idx] = super_idx
                        superclass_to_subclasses[super_idx].append(subclass)

            # Store the first-level mapping
            self.subclass_to_superclass_tensor.append(subclass_to_superclass_idx)
            self.hierarchy[0] = superclass_names  # store the list of first-level superclass names

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
                subclass_to_current_idx = [-1] * len(classnames)
                for i in range(len(classnames)):
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

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval = self.read_data(self.train_dir)
            test = self.read_data(self.test_dir)
            train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

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

    def read_data(self, image_dir):
        """Read image data from the specified directory and build multi-level labels"""
        items = []

        # Traverse each subfolder under image_dir (subclass name)
        for subclass_name in os.listdir(image_dir):
            subclass_path = os.path.join(image_dir, subclass_name)
            # Check whether it is a directory
            if os.path.isdir(subclass_path):
                # Get subclass index
                if subclass_name in self.cname2lab:
                    subclass_idx = self.cname2lab[subclass_name]

                    # Traverse image files in the subclass folder
                    for img_name in os.listdir(subclass_path):
                        img_path = os.path.join(subclass_path, img_name)
                        # Check whether it is a file rather than a directory
                        if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            # Build multi-level labels
                            label_tuple = [subclass_idx]  # subclass index

                            # Add the superclass index for each level
                            for level_tensor in self.subclass_to_superclass_tensor:
                                if 0 <= subclass_idx < len(level_tensor):
                                    label_tuple.append(level_tensor[subclass_idx])
                                else:
                                    label_tuple.append(-1)

                            # Create a Datum object, where label is a multi-level label tuple
                            item = Datum(
                                impath=img_path,  # absolute path of the image
                                label=tuple(label_tuple),  # multi-level label tuple
                                classname=subclass_name  # subclass name
                            )
                            items.append(item)
                else:
                    print(
                        f"Warning: subclass '{subclass_name}' does not have a corresponding index and will be skipped")

        return items