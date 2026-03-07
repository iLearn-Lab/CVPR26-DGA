import os
import pickle
import json
import random

from .utils import *
from .oxford_pets import OxfordPets


class StanfordCars(DatasetBase):
    dataset_dir = "stanford_cars"

    def __init__(self, root, shot, seed, subsample="all"):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        mkdir_if_missing(self.split_fewshot_dir)

        # 获取所有子文件夹名称作为类别名称
        classnames = []
        for folder_name in sorted(os.listdir(self.train_dir)):
            folder_path = os.path.join(self.train_dir, folder_name)
            if os.path.isdir(folder_path):
                classnames.append(folder_name)
        
        self._classnames = classnames
        self.cname2lab = {cname: lab for lab, cname in enumerate(classnames)}

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            # 读取训练和测试数据
            trainval = self.read_data(self.train_dir, classnames)
            test = self.read_data(self.test_dir, classnames)
            
            # 分割训练集和验证集
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

    def read_data(self, image_dir, classnames):
        """读取指定目录下的图像数据"""
        items = []
        
        # 遍历每个类别文件夹
        for class_name in classnames:
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"警告：目录 {class_dir} 不存在，跳过")
                continue
                
            class_idx = self.cname2lab[class_name]
            
            # 遍历类别文件夹中的所有图像
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                    
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    item = Datum(
                        impath=img_path,
                        label=class_idx,  # 只使用子类标签
                        classname=class_name
                    )
                    items.append(item)
        
        return items
