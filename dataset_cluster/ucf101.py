import os
import pickle
import re

from .utils import *
from .oxford_pets import OxfordPets


class UCF101(DatasetBase):

    dataset_dir = "ucf101"

    def __init__(self, root, shot, seed, subsample="all"):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            cname2lab = {}
            filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    label, classname = line.strip().split(" ")
                    label = int(label) - 1  # conver to 0-based index
                    cname2lab[classname] = label

            trainval = self.read_data(cname2lab, "ucfTrainTestlist/trainlist01.txt")
            test = self.read_data(cname2lab, "ucfTrainTestlist/testlist01.txt")
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

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []
        missing_files = []
        total_files = 0

        with open(text_file, "r") as f:
            lines = f.readlines()
            total_files = len(lines)

            for line_num, line in enumerate(lines, 1):
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)

                # 检查文件是否存在
                if not os.path.exists(impath):
                    missing_files.append({
                        'line_num': line_num,
                        'original_line': line.strip(),
                        'action': action,
                        'filename': filename,
                        'renamed_action': renamed_action,
                        'expected_path': impath
                    })
                    continue  # 跳过缺失的文件

                item = Datum(impath=impath, label=label, classname=renamed_action)
                items.append(item)

        # 输出详细的缺失文件信息
        if missing_files:
            print(f"\n=== UCF101 缺失文件报告 ===")
            print(f"文件: {text_file}")
            print(f"总文件数: {total_files}")
            print(f"缺失文件数: {len(missing_files)}")
            print(f"成功加载: {len(items)}")
            print(f"成功率: {len(items) / total_files * 100:.1f}%")

            print(f"\n=== 缺失文件详情 ===")
            for i, missing in enumerate(missing_files[:20]):  # 只显示前20个
                print(f"{i + 1}. 行号: {missing['line_num']}")
                print(f"   原始行: {missing['original_line']}")
                print(f"   动作类别: {missing['action']}")
                print(f"   重命名后: {missing['renamed_action']}")
                print(f"   期望路径: {missing['expected_path']}")
                print()

            if len(missing_files) > 20:
                print(f"... 还有 {len(missing_files) - 20} 个缺失文件未显示")

            # 保存完整的缺失文件列表到文件
            missing_log_path = os.path.join(self.dataset_dir, f"missing_files_{os.path.basename(text_file)}.txt")
            with open(missing_log_path, 'w') as f:
                f.write(f"UCF101 缺失文件报告\n")
                f.write(f"文件: {text_file}\n")
                f.write(f"总文件数: {total_files}\n")
                f.write(f"缺失文件数: {len(missing_files)}\n")
                f.write(f"成功加载: {len(items)}\n")
                f.write(f"成功率: {len(items) / total_files * 100:.1f}%\n\n")

                for missing in missing_files:
                    f.write(f"行号: {missing['line_num']}\n")
                    f.write(f"原始行: {missing['original_line']}\n")
                    f.write(f"动作类别: {missing['action']}\n")
                    f.write(f"重命名后: {missing['renamed_action']}\n")
                    f.write(f"期望路径: {missing['expected_path']}\n")
                    f.write("-" * 50 + "\n")

            print(f"\n完整的缺失文件列表已保存到: {missing_log_path}")

            # 检查目录结构
            print(f"\n=== 目录结构检查 ===")
            print(f"图像目录: {self.image_dir}")
            print(f"图像目录是否存在: {os.path.exists(self.image_dir)}")

            if os.path.exists(self.image_dir):
                subdirs = [d for d in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, d))]
                print(f"子目录数量: {len(subdirs)}")
                print(f"前10个子目录: {subdirs[:10]}")

                # 检查缺失最多的动作类别
                missing_actions = {}
                for missing in missing_files:
                    action = missing['renamed_action']
                    missing_actions[action] = missing_actions.get(action, 0) + 1

                print(f"\n=== 缺失文件最多的动作类别 ===")
                sorted_missing = sorted(missing_actions.items(), key=lambda x: x[1], reverse=True)
                for action, count in sorted_missing[:10]:
                    action_dir = os.path.join(self.image_dir, action)
                    dir_exists = os.path.exists(action_dir)
                    file_count = len(os.listdir(action_dir)) if dir_exists else 0
                    print(f"{action}: 缺失{count}个文件, 目录存在: {dir_exists}, 实际文件数: {file_count}")

        return items

