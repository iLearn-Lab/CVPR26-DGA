import os
import pickle
import re
import json
from collections import defaultdict
import random

from datasets.utils import *
from .oxford_pets import OxfordPets


class UCF101(DatasetBase):
    dataset_dir = "ucf101"

    def __init__(self, root, shot, seed, labels_list=None, model_name=None, subsample="all"):
        n_layers = len(labels_list) if labels_list else 0
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")

        # Add model_name to the split file path
        model_suffix = f"_{model_name}" if model_name else ""
        self.split_path = os.path.join(self.dataset_dir, f"split_zhou_UCF101_{n_layers}layers{model_suffix}.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot{n_layers}layers{model_suffix}")
        mkdir_if_missing(self.split_fewshot_dir)

        # Get class names and label mapping
        self.cname2lab = {}
        filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                label, classname = line.strip().split(" ")
                label = int(label) - 1  # convert to 0-based index
                self.cname2lab[classname] = label

        # Get all class names in processed format
        classnames = []
        for classname in self.cname2lab.keys():
            elements = re.findall("[A-Z][^A-Z]*", classname)
            renamed_action = "_".join(elements)
            classnames.append(renamed_action)

        # Sort by label index
        self._classnames = [None] * len(classnames)
        for classname, label in self.cname2lab.items():
            elements = re.findall("[A-Z][^A-Z]*", classname)
            renamed_action = "_".join(elements)
            self._classnames[label] = renamed_action

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
            subclass_to_superclass_idx = [-1] * len(self._classnames)
            # Store subclass names contained in each superclass
            superclass_to_subclasses = {idx: [] for idx in range(len(superclass_names))}

            for superclass, subclasses in first_level.items():
                super_idx = superclass_to_idx[superclass]
                for subclass in subclasses:
                    # Find the index of the subclass in classnames
                    for i, name in enumerate(self._classnames):
                        if name == subclass:
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
                subclass_to_current_idx = [-1] * len(self._classnames)
                for i in range(len(self._classnames)):
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
            trainval = self.read_data("ucfTrainTestlist/trainlist01.txt")
            test = self.read_data("ucfTrainTestlist/testlist01.txt")
            train, val = self.split_trainval(trainval)
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

    def read_data(self, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = self.cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)

                # Build multi-level labels
                subclass_idx = label  # subclass index
                label_tuple = [subclass_idx]

                # Add the superclass index for each level
                for level_tensor in self.subclass_to_superclass_tensor:
                    if 0 <= subclass_idx < len(level_tensor):
                        label_tuple.append(level_tensor[subclass_idx])
                    else:
                        label_tuple.append(-1)

                multi_label = tuple(label_tuple)
                item = Datum(impath=impath, label=multi_label, classname=renamed_action)
                items.append(item)

        return items

    def split_trainval(self, trainval):
        """Split the training-validation set into training and validation sets"""
        # Group by class
        tracker = defaultdict(list)
        for item in trainval:
            label = item.label[0] if isinstance(item.label, tuple) else item.label
            tracker[label].append(item)

        train, val = [], []
        for label, items in tracker.items():
            random.shuffle(items)
            n_total = len(items)
            n_train = round(n_total * 0.8)  # 80% for training
            n_val = n_total - n_train  # 20% for validation

            train.extend(items[:n_train])
            val.extend(items[n_train:])

        return train, val