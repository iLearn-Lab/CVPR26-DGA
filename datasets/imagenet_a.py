import os
import pickle
import json
from collections import OrderedDict, defaultdict

from datasets.utils import *
from .oxford_pets import OxfordPets


class ImageNetA(DatasetBase):

    def __init__(self, root, shot, seed, labels_list=None, model_name=None, subsample="all"):
        n_layers = len(labels_list) if labels_list else 1
        self.dataset_dir = os.path.join(root, "imagenet-a")
        self.image_dir = os.path.join(self.dataset_dir, "images")

        # 将 model_name 加入到预处理文件路径中
        model_suffix = f"_{model_name}" if model_name else ""
        self.preprocessed = os.path.join(self.dataset_dir, f"preprocessed{model_suffix}.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot{n_layers}layers{model_suffix}")
        mkdir_if_missing(self.split_fewshot_dir)

        # 读取 ImageNet-1K 的 classnames.txt，建立folder到classname的映射
        imagenet_dir = os.path.join(root, "imagenet")
        text_file = os.path.join(imagenet_dir, "classnames.txt")
        folder_to_classname = self.read_classnames(text_file)

        # 获取 ImageNet-1K 的所有文件夹名称（按顺序）
        imagenet_train_dir = os.path.join(imagenet_dir, "images", "train")
        folders = sorted(f.name for f in os.scandir(imagenet_train_dir) if f.is_dir())

        # ⭐ 保存完整的 1000 个类别名称列表（字符串）
        self._full_classnames = [folder_to_classname[folder] for folder in folders]
        
        # 建立类别索引，每个文件夹对应一个唯一的标签索引（与原版 ImageNet 完全一致）
        self.folder2lab = {}  # 文件夹名到标签索引的映射
        self.cname2lab = defaultdict(list)  # 类别名称到标签索引列表的映射，处理重复名称

        for label, folder in enumerate(folders):
            classname = folder_to_classname[folder]
            self.folder2lab[folder] = label
            # 将标签索引添加到对应类别名称的列表中，不覆盖
            self.cname2lab[classname].append(label)

        # 构建多层次分类关系 - 移到读取数据之前
        self.hierarchy = {}  # 存储多层次分类关系（数字索引）
        self.hierarchy_names = {}  # 存储每层的超类名称
        self.subclass_to_family_tensor = []  # 存储每层的索引映射

        if labels_list is not None:
            self._build_hierarchy(labels_list)

        # 读取或生成数据
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                data = preprocessed["data"]
        else:
            data = self.read_data(folder_to_classname)
            preprocessed = {"data": data}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 所有数据同时作为 train 和 test
        train = data
        test = data

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

    # ⭐ 覆盖 get_lab2cname 方法
    def get_lab2cname(self, data_source):
        """返回完整的 ImageNet-1K 标签到类别名称的映射"""
        # 构建标签到类别名称的映射（使用完整的 1000 个类别）
        lab2cname = {i: self._full_classnames[i] for i in range(len(self._full_classnames))}
        
        # 返回映射和完整的类别名称列表
        return lab2cname, self._full_classnames

    # ⭐ 覆盖 classnames 属性（双重保险）
    @property
    def classnames(self):
        """返回完整的 ImageNet-1K 的 1000 个类别名称（字符串列表）"""
        return self._full_classnames

    # ⭐ 覆盖 num_classes 属性
    @property
    def num_classes(self):
        """返回 1000（ImageNet-1K 的类别数）"""
        return len(self._full_classnames)

    def _build_hierarchy(self, labels_list):
        """构建多层次分类关系（与原版 ImageNet 完全一致）"""
        n_layers = len(labels_list)

        # 处理第一层超类与子类的关系
        with open(labels_list[0], 'r') as f:
            first_level = json.load(f)

        # 为第一层超类建立索引（按照JSON文件中的顺序）
        superclass_names = list(first_level.keys())
        superclass_to_idx = {name: idx for idx, name in enumerate(superclass_names)}

        # 建立子类到第一层超类的映射
        subclass_to_superclass_idx = [-1] * len(self._full_classnames)
        # 存储每个超类包含的子类名称
        superclass_to_subclasses = {idx: [] for idx in range(len(superclass_names))}

        for superclass, subclasses in first_level.items():
            super_idx = superclass_to_idx[superclass]
            for subclass in subclasses:
                # 查找所有匹配的子类标签索引（处理重复名称）
                if subclass in self.cname2lab:
                    for class_idx in self.cname2lab[subclass]:
                        subclass_to_superclass_idx[class_idx] = super_idx
                        superclass_to_subclasses[super_idx].append(subclass)

        # 存储第一层的映射
        self.subclass_to_family_tensor.append(subclass_to_superclass_idx)
        self.hierarchy[0] = superclass_names

        # 构建第一层超类的名称（使用子类名称）
        superclass_names_dict = {}
        for idx, subclasses in superclass_to_subclasses.items():
            if subclasses:
                # 去重但保持顺序
                unique_subclasses = []
                seen = set()
                for sc in subclasses:
                    if sc not in seen:
                        unique_subclasses.append(sc)
                        seen.add(sc)
                combined_name = " or ".join(unique_subclasses)
                superclass_names_dict[idx] = combined_name
            else:
                superclass_names_dict[idx] = f"layer1_{idx}"
        self.hierarchy_names[0] = superclass_names_dict

        # 处理更高层次的超类关系
        prev_superclass_names = superclass_names
        prev_superclass_to_subclasses = superclass_to_subclasses

        for level in range(1, n_layers):
            with open(labels_list[level], 'r') as f:
                current_level = json.load(f)

            # 为当前层超类建立索引
            current_superclass_names = list(current_level.keys())
            current_superclass_to_idx = {name: idx for idx, name in enumerate(current_superclass_names)}

            # 建立上一层超类到当前层超类的映射
            prev_to_current_idx = [-1] * len(prev_superclass_names)
            # 存储每个当前层超类包含的所有基础子类
            current_superclass_to_subclasses = {idx: [] for idx in range(len(current_superclass_names))}

            for superclass, subclasses in current_level.items():
                super_idx = current_superclass_to_idx[superclass]
                for subclass in subclasses:
                    if subclass in prev_superclass_names:
                        prev_idx = prev_superclass_names.index(subclass)
                        prev_to_current_idx[prev_idx] = super_idx

                        # 将上一层超类包含的所有子类添加到当前层超类
                        if prev_idx in prev_superclass_to_subclasses:
                            current_superclass_to_subclasses[super_idx].extend(
                                prev_superclass_to_subclasses[prev_idx]
                            )

            # 构建子类到当前层超类的映射
            subclass_to_current_idx = [-1] * len(self._full_classnames)
            for i in range(len(self._full_classnames)):
                prev_level_idx = self.subclass_to_family_tensor[level - 1][i]
                if prev_level_idx != -1 and prev_level_idx < len(prev_to_current_idx):
                    subclass_to_current_idx[i] = prev_to_current_idx[prev_level_idx]

            # 存储当前层的映射
            self.subclass_to_family_tensor.append(subclass_to_current_idx)
            self.hierarchy[level] = current_superclass_names

            # 构建当前层超类的名称
            current_names_dict = {}
            for idx, subclasses in current_superclass_to_subclasses.items():
                if subclasses:
                    unique_subclasses = []
                    seen = set()
                    for sc in subclasses:
                        if sc not in seen:
                            unique_subclasses.append(sc)
                            seen.add(sc)
                    combined_name = " or ".join(unique_subclasses)
                    current_names_dict[idx] = combined_name
                else:
                    current_names_dict[idx] = f"layer{level + 1}_{idx}"
            self.hierarchy_names[level] = current_names_dict

            # 更新变量，用于下一次迭代
            prev_superclass_names = current_superclass_names
            prev_superclass_to_subclasses = current_superclass_to_subclasses

        # 构建 classes_f 结构
        classes_f = {}
        for level in range(n_layers):
            classes_f[level] = {}
            for idx, name in self.hierarchy_names[level].items():
                classes_f[level][idx] = name

        # 如果只有一层，添加一个虚拟的第二层
        if n_layers == 1:
            classes_f[1] = {0: "layer2_0"}
            # 将所有子类映射到这个虚拟的第二层
            all_to_single = [0] * len(self._full_classnames)
            self.subclass_to_family_tensor.append(all_to_single)

        # 将 classes_f 存储为实例变量
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

    def read_data(self, folder_to_classname):
        """读取 ImageNet-A 的图像数据，使用 ImageNet-1K 的类别名称和索引"""
        folders = sorted(f.name for f in os.scandir(self.image_dir) if f.is_dir())
        items = []

        print(f"\n{'=' * 80}")
        print(f"Reading ImageNet-A Data")
        print(f"{'=' * 80}")
        print(f"Found {len(folders)} folders in {self.image_dir}")

        # 统计每个类别的样本数
        label_counts = {}
        skipped_folders = []

        for folder_idx, folder in enumerate(folders):
            # 检查该文件夹是否在 ImageNet-1K 中
            if folder not in self.folder2lab:
                skipped_folders.append(folder)
                continue

            # 使用 ImageNet-1K 的标签和类别名称
            label = self.folder2lab[folder]
            classname = folder_to_classname[folder]

            imnames = listdir_nohidden(os.path.join(self.image_dir, folder))

            for imname in imnames:
                impath = os.path.join(self.image_dir, folder, imname)

                # 构建多层次标签（与原版 ImageNet 完全一致）
                if hasattr(self, 'subclass_to_family_tensor') and self.subclass_to_family_tensor:
                    label_tuple = [label]  # 子类索引

                    # 添加每一层的超类索引
                    for level_tensor in self.subclass_to_family_tensor:
                        if 0 <= label < len(level_tensor):
                            label_tuple.append(level_tensor[label])
                        else:
                            label_tuple.append(-1)

                    item = Datum(impath=impath, label=tuple(label_tuple), classname=classname)
                else:
                    # 如果没有多层次标签，使用单一标签
                    item = Datum(impath=impath, label=label, classname=classname)

                items.append(item)

            label_counts[label] = len(imnames)

        # 打印统计信息
        print(f"\nTotal images loaded: {len(items)}")
        print(f"Number of classes: {len(label_counts)}")
        if skipped_folders:
            print(f"Skipped {len(skipped_folders)} folders not in ImageNet-1K")
        print(f"{'=' * 80}\n")

        return items
