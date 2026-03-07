import os
import pickle
import json
from collections import OrderedDict, defaultdict

from datasets.utils import *
from .oxford_pets import OxfordPets


class ImageNet(DatasetBase):

    def __init__(self, root, shot, seed, labels_list=None, model_name=None, subsample="all"):
        n_layers = len(labels_list) if labels_list else 1
        self.dataset_dir = os.path.join(root, "imagenet")
        self.image_dir = os.path.join(self.dataset_dir, "images")

        # Add model_name to the preprocessed file path
        model_suffix = f"_{model_name}" if model_name else ""
        self.preprocessed = os.path.join(self.dataset_dir, f"preprocessed{model_suffix}.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot{n_layers}layers{model_suffix}")
        mkdir_if_missing(self.split_fewshot_dir)

        # Read classnames.txt and build the mapping from folder to classname
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        folder_to_classname = self.read_classnames(text_file)

        # Get all folder names in order so that duplicate class names can be handled
        train_dir = os.path.join(self.image_dir, "train")
        folders = sorted(f.name for f in os.scandir(train_dir) if f.is_dir())

        # Build class indices, where each folder corresponds to a unique label index
        self._classnames = []
        self.folder2lab = {}  # mapping from folder name to label index
        self.cname2lab = defaultdict(list)  # mapping from class name to a list of label indices for handling duplicate names

        for label, folder in enumerate(folders):
            classname = folder_to_classname[folder]
            self._classnames.append(classname)
            self.folder2lab[folder] = label
            # Append the label index to the corresponding classname list instead of overwriting
            self.cname2lab[classname].append(label)

        # Build multi-level hierarchy relationships before reading data
        self.hierarchy = {}  # store multi-level hierarchy relationships (index-based)
        self.hierarchy_names = {}  # store superclass names at each level
        self.subclass_to_superclass_tensor = []  # store index mappings for each level

        if labels_list is not None:
            self._build_hierarchy(labels_list)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            train = self.read_data(folder_to_classname, "train")
            test = self.read_data(folder_to_classname, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = shot
        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    def _build_hierarchy(self, labels_list):
        """Build multi-level hierarchy relationships"""
        n_layers = len(labels_list)

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
                # Find all matching subclass label indices to handle duplicate names
                if subclass in self.cname2lab:
                    for class_idx in self.cname2lab[subclass]:
                        subclass_to_superclass_idx[class_idx] = super_idx
                        superclass_to_subclasses[super_idx].append(subclass)

        # Store the first-level mapping
        self.subclass_to_superclass_tensor.append(subclass_to_superclass_idx)
        self.hierarchy[0] = superclass_names  # store the list of first-level superclass names

        # Build names for first-level superclasses using subclass names
        superclass_names_dict = {}
        for idx, subclasses in superclass_to_subclasses.items():
            if subclasses:  # ensure subclasses exist
                # Remove duplicates while preserving order
                unique_subclasses = []
                seen = set()
                for sc in subclasses:
                    if sc not in seen:
                        unique_subclasses.append(sc)
                        seen.add(sc)
                combined_name = " or ".join(unique_subclasses)
                superclass_names_dict[idx] = combined_name
            else:
                superclass_names_dict[idx] = f"layer1_{idx}"  # default name
        self.hierarchy_names[0] = superclass_names_dict

        # Process higher-level superclass relationships
        prev_superclass_names = superclass_names  # layer1_0, layer1_1, ...
        prev_superclass_to_subclasses = superclass_to_subclasses

        for level in range(1, n_layers):
            with open(labels_list[level], 'r') as f:
                current_level = json.load(f)

            # Build indices for current-level superclasses
            current_superclass_names = list(current_level.keys())
            current_superclass_to_idx = {name: idx for idx, name in enumerate(current_superclass_names)}

            # Build mapping from previous-level superclasses to current-level superclasses
            prev_to_current_idx = [-1] * len(prev_superclass_names)
            # Store all base subclasses contained in each current-level superclass
            current_superclass_to_subclasses = {idx: [] for idx in range(len(current_superclass_names))}

            for superclass, subclasses in current_level.items():
                super_idx = current_superclass_to_idx[superclass]
                for subclass in subclasses:
                    if subclass in prev_superclass_names:
                        prev_idx = prev_superclass_names.index(subclass)
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
                    unique_subclasses = []
                    seen = set()
                    for sc in subclasses:
                        if sc not in seen:
                            unique_subclasses.append(sc)
                            seen.add(sc)
                    combined_name = " or ".join(unique_subclasses)
                    current_names_dict[idx] = combined_name
                else:
                    current_names_dict[idx] = f"layer{level + 1}_{idx}"  # default name
            self.hierarchy_names[level] = current_names_dict

            # Update variables for the next iteration
            prev_superclass_names = current_superclass_names
            prev_superclass_to_subclasses = current_superclass_to_subclasses

        # Build classes_f structure
        classes_f = {}
        for level in range(n_layers):
            classes_f[level] = {}
            for idx, name in self.hierarchy_names[level].items():
                classes_f[level][idx] = name

        # If there is only one layer, add a dummy second layer
        if n_layers == 1:
            classes_f[1] = {0: "layer2_0"}
            # Map all subclasses to this dummy second layer
            all_to_single = [0] * len(self._classnames)
            self.subclass_to_superclass_tensor.append(all_to_single)

        # Store classes_f as an instance variable if needed
        self.classes_f = classes_f

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, folder_to_classname, split_dir):
        """Read image data from the specified directory and build multi-level labels"""
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for folder in folders:
            if folder in self.folder2lab:
                label = self.folder2lab[folder]
                classname = folder_to_classname[folder]
                imnames = listdir_nohidden(os.path.join(split_dir, folder))

                for imname in imnames:
                    impath = os.path.join(split_dir, folder, imname)

                    # Build multi-level labels
                    if hasattr(self, 'subclass_to_superclass_tensor') and self.subclass_to_superclass_tensor:
                        label_tuple = [label]  # subclass index

                        # Add the superclass index of each level
                        for level_tensor in self.subclass_to_superclass_tensor:
                            if 0 <= label < len(level_tensor):
                                label_tuple.append(level_tensor[label])
                            else:
                                label_tuple.append(-1)

                        item = Datum(impath=impath, label=tuple(label_tuple), classname=classname)
                    else:
                        # If there are no multi-level labels, use a single label
                        item = Datum(impath=impath, label=label, classname=classname)

                    items.append(item)

        return items