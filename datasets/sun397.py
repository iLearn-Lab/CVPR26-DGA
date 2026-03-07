import os
import pickle
import json
from collections import defaultdict
import random

from datasets.utils import *
from .oxford_pets import OxfordPets


class SUN397(DatasetBase):
    dataset_dir = "sun397"

    def __init__(self, root, shot, seed, labels_list=None, model_name=None, subsample="all"):
        n_layers = len(labels_list) if labels_list else 0
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")

        # Add model_name to the split file path
        model_suffix = f"_{model_name}" if model_name else ""
        self.split_path = os.path.join(self.dataset_dir, f"split_zhou_SUN397_{n_layers}layers{model_suffix}.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot_{n_layers}layers{model_suffix}")
        mkdir_if_missing(self.split_fewshot_dir)

        # Read original class names
        classnames = []
        with open(os.path.join(self.dataset_dir, "ClassName.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()[1:]  # remove first /
                # Process the class name: remove everything before and including the second slash, then reorder
                processed_name = self.process_classname(line)
                classnames.append(processed_name)

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
                    # Keep class names in layer0 unchanged, without replacing underscores
                    processed_subclass = subclass
                    # Find the index of the subclass in classnames
                    for i, name in enumerate(classnames):
                        if name == processed_subclass:
                            subclass_to_superclass_idx[i] = super_idx
                            superclass_to_subclasses[super_idx].append(name)
                            break

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
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            # Build the original cname2lab mapping using the original processing method
            original_classnames = []
            with open(os.path.join(self.dataset_dir, "ClassName.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()[1:]  # remove /
                    original_classnames.append(line)

            cname2lab = {c: i for i, c in enumerate(original_classnames)}
            trainval = self.read_data(cname2lab, "Training_01.txt", classnames)
            test = self.read_data(cname2lab, "Testing_01.txt", classnames)
            train, val = OxfordPets.split_trainval(trainval)
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
        self._classnames = classnames

    def process_classname(self, original_name):
        """
        Process the class name:
        For example:
        "a/amusement_park" -> "amusement_park"
        "a/apartment_building/outdoor" -> "outdoor apartment_building"
        1. Remove the first part (such as 'a', 'b', etc.)
        2. If multiple parts remain, move the last part to the front
        3. Replace slashes with spaces while preserving underscores
        """
        parts = original_name.split('/')

        if len(parts) >= 2:
            # Remove the first part (such as 'a', 'b', etc.) and keep the content starting from the second slash
            remaining_parts = parts[1:]

            if len(remaining_parts) >= 2:
                # If multiple parts remain, move the last part to the front
                # For example: ['apartment_building', 'outdoor'] -> ['outdoor', 'apartment_building']
                reordered_parts = [remaining_parts[-1]] + remaining_parts[:-1]
            else:
                # If only one part remains, use it directly
                # For example: ['amusement_park'] -> ['amusement_park']
                reordered_parts = remaining_parts

            # Join with spaces while preserving underscores
            processed_name = " ".join(reordered_parts)
        else:
            # If only one part remains after splitting, process it directly
            processed_name = " ".join(parts)

        return processed_name

    def read_data(self, cname2lab, text_file, processed_classnames):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                original_classname = os.path.dirname(imname)
                label = cname2lab[original_classname]
                impath = os.path.join(self.image_dir, imname)

                # Use the processed class name
                processed_classname = processed_classnames[label]

                # Build multi-level labels
                if hasattr(self, 'subclass_to_superclass_tensor') and self.subclass_to_superclass_tensor:
                    label_tuple = [label]
                    # Add the superclass index for each level
                    for level_tensor in self.subclass_to_superclass_tensor:
                        if 0 <= label < len(level_tensor):
                            label_tuple.append(level_tensor[label])
                        else:
                            label_tuple.append(-1)
                    label = tuple(label_tuple)

                item = Datum(impath=impath, label=label, classname=processed_classname)
                items.append(item)

        return items