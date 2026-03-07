import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from tqdm import tqdm

from torchvision.transforms import (
    Compose, Resize, Lambda, ToTensor, CenterCrop,
    RandomResizedCrop, RandomHorizontalFlip, InterpolationMode
)

from cfg import *
from tools import *
from datasets import *
from methods.vp import PaddingVR
from clip import clip


class VisualGranularity(nn.Module):
    """Visual granularity module"""

    def __init__(self, num_experts=3, n_crops=5, input_size=192, entropy_threshold=0.9):
        super().__init__()
        self.num_experts = num_experts
        self.n_crops = n_crops
        self.input_size = input_size
        self.entropy_threshold = entropy_threshold

        self.visual_reprograms = nn.ModuleList()
        self.text_projections = nn.ModuleList()

        for _ in range(num_experts):
            vr = PaddingVR(224, input_size)
            proj = TextProjection()
            self.visual_reprograms.append(vr)
            self.text_projections.append(proj)

    def random_crop_and_resize(self, x, crop_size):
        """Randomly crop and resize"""
        batch_size, c, h, w = x.shape
        device = x.device

        crop_h = crop_w = crop_size

        top = torch.randint(0, h - crop_h + 1, (batch_size,), device=device)
        left = torch.randint(0, w - crop_w + 1, (batch_size,), device=device)

        cropped = torch.zeros(batch_size, c, crop_h, crop_w, device=device)
        for i in range(batch_size):
            cropped[i] = x[i, :, top[i]:top[i] + crop_h, left[i]:left[i] + crop_w]

        cropped = F.interpolate(
            cropped,
            size=(self.input_size, self.input_size),
            mode='bilinear',
            align_corners=False
        )
        return cropped

    def compute_entropy(self, logits):
        """Compute entropy of logits"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def entropy_weighted_fusion(self, logits_list):
        """Entropy-weighted fusion"""
        if len(logits_list) == 0:
            return None

        if len(logits_list) == 1:
            return logits_list[0]

        entropies = [self.compute_entropy(logits) for logits in logits_list]

        valid_indices = []
        for i, entropy in enumerate(entropies):
            if torch.mean(entropy) < self.entropy_threshold:
                valid_indices.append(i)

        if len(valid_indices) == 0:
            return torch.stack(logits_list).mean(dim=0)

        valid_logits = [logits_list[i] for i in valid_indices]
        valid_entropies = [entropies[i] for i in valid_indices]

        weights = []
        for entropy in valid_entropies:
            weight = 1.0 - torch.mean(entropy)
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)

        fused_logits = torch.zeros_like(valid_logits[0])
        for logits, weight in zip(valid_logits, weights):
            fused_logits += weight * logits

        return fused_logits

    def forward_single_granularity(
        self,
        x,
        granularity_idx,
        clip_model,
        txt_emb_desattr,
        txt_emb_distattr,
        alpha,
        num_attr,
        k,
        superclass_vrs=None
    ):
        """Forward of one visual granularity"""
        if granularity_idx == 0:
            if superclass_vrs is not None:
                program_list = []
                for superclass_vr in superclass_vrs:
                    program_list.append(superclass_vr.program)
                program_list.append(self.visual_reprograms[granularity_idx].program)
                avg_program = torch.stack(program_list).mean(dim=0)

                x_padded = F.pad(
                    x,
                    (
                        self.visual_reprograms[granularity_idx].l_pad,
                        self.visual_reprograms[granularity_idx].r_pad,
                        self.visual_reprograms[granularity_idx].l_pad,
                        self.visual_reprograms[granularity_idx].r_pad
                    ),
                    value=0
                )
                x_vr = x_padded + torch.sigmoid(avg_program) * self.visual_reprograms[granularity_idx].mask
                if self.visual_reprograms[granularity_idx].norm:
                    x_vr = self.visual_reprograms[granularity_idx].normalize(x_vr)
            else:
                x_vr = self.visual_reprograms[granularity_idx](x)

            x_emb = clip_model.encode_image(x_vr)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            exp = clip_model.logit_scale.exp()

            desattr_logits = getLogits(
                num_attr, exp, x_emb,
                self.text_projections[granularity_idx](txt_emb_desattr), k=k
            )
            distattr_logits = getLogits(
                num_attr, exp, x_emb,
                self.text_projections[granularity_idx](txt_emb_distattr), k=k
            )
            logits = desattr_logits * alpha + distattr_logits * (1 - alpha)

            return logits, x_emb

        else:
            crop_logits = []
            crop_features = []

            base_crop_size = self.input_size - 32 * granularity_idx
            crop_size = max(base_crop_size, 64)

            for _ in range(self.n_crops):
                x_crop = self.random_crop_and_resize(x, crop_size)

                x_vr = self.visual_reprograms[granularity_idx](x_crop)

                x_emb = clip_model.encode_image(x_vr)
                x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
                exp = clip_model.logit_scale.exp()

                desattr_logits = getLogits(
                    num_attr, exp, x_emb,
                    self.text_projections[granularity_idx](txt_emb_desattr), k=k
                )
                distattr_logits = getLogits(
                    num_attr, exp, x_emb,
                    self.text_projections[granularity_idx](txt_emb_distattr), k=k
                )
                logits = desattr_logits * alpha + distattr_logits * (1 - alpha)

                crop_logits.append(logits)
                crop_features.append(x_emb)

            fused_logits = self.entropy_weighted_fusion(crop_logits)
            fused_features = torch.stack(crop_features).mean(dim=0)

            return fused_logits, fused_features


class SemanticGranularity(nn.Module):
    """Semantic granularity module"""

    def __init__(self, input_size=192):
        super().__init__()
        self.input_size = input_size
        self.visual_reprogram = PaddingVR(224, input_size)
        self.text_projection = TextProjection()

    def forward(self, x, clip_model, txt_emb_desattr, txt_emb_distattr, alpha, num_attr, k):
        x_vr = self.visual_reprogram(x)
        x_emb = clip_model.encode_image(x_vr)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        exp = clip_model.logit_scale.exp()

        desattr_logits = getLogits(
            num_attr, exp, x_emb,
            self.text_projection(txt_emb_desattr), k=k
        )
        distattr_logits = getLogits(
            num_attr, exp, x_emb,
            self.text_projection(txt_emb_distattr), k=k
        )
        logits = desattr_logits * alpha + distattr_logits * (1 - alpha)

        return logits, x_emb

    def HKP(self, superclass_logits, subclass_to_superclass_tensor, num_subclasses, subclass_logits, eps=1e-12):
        """
        HKP with fully-parallel redistribution (no Python loop).

        We rescale subclass logits within each superclass so that:
            mean(subclass_logits_scaled within a superclass) == corresponding superclass logit
        """
        device = superclass_logits.device
        B, num_superclasses = superclass_logits.shape

        if not isinstance(subclass_to_superclass_tensor, torch.Tensor):
            subclass_to_superclass_tensor = torch.tensor(
                subclass_to_superclass_tensor, device=device, dtype=torch.long
            )
        else:
            subclass_to_superclass_tensor = subclass_to_superclass_tensor.to(
                device=device, dtype=torch.long
            )

        expanded_logits = superclass_logits[:, subclass_to_superclass_tensor]

        sum_sub = torch.zeros(B, num_superclasses, device=device, dtype=subclass_logits.dtype)
        sum_sub.scatter_add_(
            dim=1,
            index=subclass_to_superclass_tensor.unsqueeze(0).expand(B, -1),
            src=subclass_logits
        )

        ones = torch.ones(1, num_subclasses, device=device, dtype=subclass_logits.dtype)
        count_sub = torch.zeros(1, num_superclasses, device=device, dtype=subclass_logits.dtype)
        count_sub.scatter_add_(
            dim=1,
            index=subclass_to_superclass_tensor.unsqueeze(0),
            src=ones
        )
        count_sub = count_sub.expand(B, -1)

        mean_sub = sum_sub / (count_sub + eps)

        mean_sub_per_subclass = mean_sub[:, subclass_to_superclass_tensor]
        super_logit_per_subclass = expanded_logits

        scale_per_subclass = super_logit_per_subclass / (mean_sub_per_subclass + eps)

        hkp_logits = subclass_logits * scale_per_subclass

        return hkp_logits


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--batchsize', type=int, default=64)
    p.add_argument(
        '--dataset',
        choices=[
            'caltech101', 'dtd', 'eurosat', 'fgvc', 'food101',
            'oxford_flowers', 'oxford_pets', 'stanford_cars',
            'sun397', 'ucf101', 'resisc45', 'imagenet'
        ],
        default='dtd'
    )
    p.add_argument(
        '--model_name',
        type=str,
        default='ViT-B/16',
        choices=['ViT-B/16', 'ViT-B/32', 'RN50', 'RN101'],
        help='CLIP backbone model name'
    )
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--num_attr', type=int, default=20)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=40)
    p.add_argument('--input_size', type=int, default=192, help='224x224 images with VR frame size = 16')
    p.add_argument('--shot', type=int, default=16)
    p.add_argument('--test', action='store_true', default=False, help='directly test with checkpoint')

    p.add_argument('--num_sg_layers', type=int, default=3, help='number of superclass layers')
    p.add_argument('--mix_weight', type=float, default=0.7, help='weight for logits mixture')
    p.add_argument('--proj_lr', type=float, default=0.01, help='learning rate for text projection')

    p.add_argument('--num_vg_scales', type=int, default=2, help='number of visual granularity scales')
    p.add_argument('--n_crops', type=int, default=5, help='number of crops per visual granularity scale')
    p.add_argument('--entropy_threshold', type=float, default=0.9, help='entropy threshold for fusion')

    args = p.parse_args()
    args.num_vg_scales += 1
    seed = args.seed
    print("Setting fixed seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model_tag = args.model_name.replace("/", "-")
    exp = f"results_seed{seed}/dga"
    save_path = os.path.join(
        exp,
        f"{args.dataset}_{model_tag}_shot{args.shot}_k{args.k}_a{args.alpha}_s{args.seed}_e{args.num_vg_scales}"
    )
    mkdir_if_missing(save_path)

    clip_model, _ = clip.load(args.model_name, device=device)
    convert_models_to_fp32(clip_model)
    clip_model.eval()
    clip_model.requires_grad_(False)

    train_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])

    test_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(args.input_size),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])

    if args.shot == 1:
        bs = 16
    else:
        bs = args.batchsize

    labels_list = []
    if args.model_name == "ViT-B/16":
        label_modelname = "vit_b_16"
    elif args.model_name == "ViT-B/32":
        label_modelname = "vit_b_32"
    else:
        label_modelname = args.model_name.replace("/", "-")

    label_path = os.path.join("./labels_clip", label_modelname, args.dataset)

    for i in range(args.num_sg_layers):
        label_path_i = os.path.join(label_path, f"layer{i + 1}.json")
        labels_list.append(label_path_i)

    trainloader, testloader, classes, classes_hierarchy, subclass_to_superclass = build_loader(
        args.dataset,
        DOWNSTREAM_PATH,
        train_process,
        test_process,
        batch_size=bs,
        shot=args.shot,
        labels_list=labels_list
    )

    classes_lists = []
    for level in sorted(classes_hierarchy.keys()):
        level_classes = []
        for class_idx in sorted(classes_hierarchy[level].keys()):
            class_name = classes_hierarchy[level][class_idx]
            level_classes.append(class_name)
        classes_lists.append(level_classes)

    txt_emb_desattr = clip_attr_classifier(
        classes, clip_model, 'attributes/gpt3/' + args.dataset + '_des.json',
        num_attr=args.num_attr
    )
    txt_emb_distattr = clip_attr_classifier(
        classes, clip_model, 'attributes/gpt3/' + args.dataset + '_dist.json',
        num_attr=args.num_attr
    )

    txt_emb_desattr_sup = []
    txt_emb_distattr_sup = []
    for i in range(args.num_sg_layers):
        txt_emb_desattr_sup_i, txt_emb_distattr_sup_i = aggregate_subclass_embeddings_to_superclass(
            args.num_attr,
            i,
            txt_emb_desattr,
            txt_emb_distattr,
            subclass_to_superclass[i]
        )
        txt_emb_desattr_sup_i = F.normalize(txt_emb_desattr_sup_i, dim=-1)
        txt_emb_distattr_sup_i = F.normalize(txt_emb_distattr_sup_i, dim=-1)
        txt_emb_desattr_sup.append(txt_emb_desattr_sup_i)
        txt_emb_distattr_sup.append(txt_emb_distattr_sup_i)

    vg_module = VisualGranularity(
        num_experts=args.num_vg_scales,
        n_crops=args.n_crops,
        input_size=args.input_size,
        entropy_threshold=args.entropy_threshold
    ).to(device)

    sg_module = nn.ModuleList()
    for i in range(args.num_sg_layers):
        sg_module_i = SemanticGranularity(input_size=192).to(device)
        sg_module.append(sg_module_i)

    superclass_names_list = []
    for level in sorted(classes_hierarchy.keys()):
        level_classes = []
        for class_idx in sorted(classes_hierarchy[level].keys()):
            class_name = classes_hierarchy[level][class_idx]
            level_classes.append(class_name)
        superclass_names_list.append(level_classes)

    def network_VG(x):
        """Subclass prediction with visual granularity branch"""
        vg_logits_list = []
        vg_features_list = []

        superclass_vrs = [sg_module[i].visual_reprogram for i in range(args.num_sg_layers)]

        for vg_scale in range(args.num_vg_scales):
            logits, features = vg_module.forward_single_granularity(
                x,
                vg_scale,
                clip_model,
                txt_emb_desattr,
                txt_emb_distattr,
                args.alpha,
                args.num_attr,
                args.k,
                superclass_vrs=superclass_vrs
            )
            vg_logits_list.append(logits)
            vg_features_list.append(features)

        final_logits = torch.stack(vg_logits_list).mean(dim=0)
        final_features = torch.stack(vg_features_list).mean(dim=0)

        return final_logits, final_features

    def network_SG(x, layer_idx):
        """Superclass prediction with semantic granularity branch"""
        logits, features = sg_module[layer_idx].forward(
            x,
            clip_model,
            txt_emb_desattr_sup[layer_idx],
            txt_emb_distattr_sup[layer_idx],
            args.alpha,
            args.num_attr * 2 ** (layer_idx + 1),
            args.k
        )
        return logits, features

    checkpoint_path = os.path.join(save_path, 'best_model.pth')

    if args.test:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)

        vg_module.load_state_dict(checkpoint["vg_module"])
        for layer_idx in range(args.num_sg_layers):
            sg_module[layer_idx].load_state_dict(checkpoint[f"expert_moe_father_{layer_idx}"])

        print(f"Loaded checkpoint: {checkpoint_path}, best accuracy: {checkpoint['best_acc']}")

    else:
        all_params = []

        all_params.extend([
            {'params': vg_module.visual_reprograms.parameters(), 'lr': args.lr},
            {'params': vg_module.text_projections.parameters(), 'lr': args.proj_lr},
        ])

        for i in range(args.num_sg_layers):
            all_params.extend([
                {'params': sg_module[i].visual_reprogram.parameters(), 'lr': args.lr / 2},
                {'params': sg_module[i].text_projection.parameters(), 'lr': args.proj_lr},
            ])

        optimizer = torch.optim.SGD(all_params, momentum=0.9)
        t_max = args.epoch * len(trainloader)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    def test_model(testloader, gamma=0.8, is_test=False, save_path=None):
        vg_module.eval()
        for expert_moe_f in sg_module:
            expert_moe_f.eval()

        total_num = 0
        true_num = 0
        true_num_sg_list = [0] * args.num_sg_layers
        true_num_mix = 0

        all_preds_subclass = []
        all_preds_sg_list = [[] for _ in range(args.num_sg_layers)]
        all_preds_mix = []
        all_targets_subclass = []
        all_targets_sg_list = [[] for _ in range(args.num_sg_layers)]

        if is_test:
            import matplotlib.pyplot as plt

            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            layer_features_list = [[] for _ in range(args.num_sg_layers)]
            layer_labels_list = [[] for _ in range(args.num_sg_layers)]

            subclass_features = []
            subclass_labels = []

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(device)
            y_subclass = y[0].to(device)

            with torch.no_grad():
                logits_vg, image_features = network_VG(x)

                if is_test:
                    subclass_features.append(image_features.cpu())
                    subclass_labels.extend(y_subclass.cpu().numpy())

                expanded_logits_list = []

                for layer_idx in range(args.num_sg_layers):
                    y_super = y[layer_idx + 1].to(device)
                    logits_sg, features_sg = network_SG(x, layer_idx)

                    if is_test:
                        layer_features_list[layer_idx].append(features_sg.cpu())
                        layer_labels_list[layer_idx].extend(y_super.cpu().numpy())

                    pred_sg = torch.argmax(logits_sg, 1)
                    true_num_sg_list[layer_idx] += pred_sg.eq(y_super).float().sum().item()
                    all_preds_sg_list[layer_idx].extend(pred_sg.cpu().numpy())
                    all_targets_sg_list[layer_idx].extend(y_super.cpu().numpy())

                    subclass_to_superclass_i = subclass_to_superclass[layer_idx]
                    expanded_logits = sg_module[layer_idx].HKP(
                        F.log_softmax(logits_sg, dim=-1),
                        subclass_to_superclass_i,
                        len(classes),
                        logits_vg
                    )
                    expanded_logits_list.append(expanded_logits)

                avg_expanded_logits = torch.stack(expanded_logits_list).mean(dim=0)
                logits_mix = gamma * F.log_softmax(logits_vg, dim=-1) + (1 - gamma) * avg_expanded_logits

                pred_subclass = torch.argmax(logits_vg, 1)
                pred_mix = torch.argmax(logits_mix, 1)

                total_num += y_subclass.size(0)
                true_num += pred_subclass.eq(y_subclass).float().sum().item()
                true_num_mix += pred_mix.eq(y_subclass).float().sum().item()

                all_preds_subclass.extend(pred_subclass.cpu().numpy())
                all_preds_mix.extend(pred_mix.cpu().numpy())
                all_targets_subclass.extend(y_subclass.cpu().numpy())

        acc = true_num / total_num
        acc_sg_list = [num_sg / total_num for num_sg in true_num_sg_list]
        acc_mix = true_num_mix / total_num

        return acc, acc_sg_list, acc_mix

    if args.test:
        cm_save_path = "./visualization_result_moe"
        cm_save_path = os.path.join(cm_save_path, args.dataset, str(args.num_vg_scales))
        mkdir_if_missing(cm_save_path)

        print("Entering test mode. Loading the best checkpoint for evaluation...")
        acc, acc_sg_list, acc_mix = test_model(
            testloader,
            is_test=True,
            save_path=cm_save_path,
            gamma=args.mix_weight
        )

        print(f"\n{'=' * 60}")
        print("Test Results Summary")
        print(f"{'=' * 60}")
        print(f"Number of superclass layers: {args.num_sg_layers}")
        print(f"Number of visual granularity scales: {args.num_vg_scales}")
        print(f"Number of random crops: {args.n_crops}")
        print(f"Entropy threshold: {args.entropy_threshold}")
        print(f"Subclass accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Final accuracy: {acc_mix:.4f} ({acc_mix * 100:.2f}%)")

        for i, acc_sg in enumerate(acc_sg_list):
            print(f"Superclass accuracy at level {i + 1}: {acc_sg:.4f} ({acc_sg * 100:.2f}%)")

        print(f"{'=' * 60}")

    else:
        vg_module.train()
        for expert_moe_f in sg_module:
            expert_moe_f.train()

        best_acc_mix = 0.
        progress_bar = tqdm(total=args.epoch, desc='Training', leave=True)

        for epoch in range(args.epoch):

            total_num = 0
            loss_sum = 0
            true_num = 0
            true_num_sg_list = [0] * args.num_sg_layers
            true_num_mix = 0

            for i, (x, y) in enumerate(trainloader):
                x = x.to(device)

                y_subclass = y[0].to(device)
                batch_size = y_subclass.size(0)

                optimizer.zero_grad()

                with autocast(device_type='cuda'):
                    logits_vg, subclass_features = network_VG(x)

                    expanded_logits_list = []
                    superclass_losses = []

                    for layer_idx in range(args.num_sg_layers):
                        y_super = y[layer_idx + 1].to(device)

                        logits_sg, features_sg = network_SG(x, layer_idx)
                        loss_sg = F.cross_entropy(logits_sg, y_super, reduction='mean')

                        superclass_losses.append(loss_sg)

                        subclass_to_superclass_i = subclass_to_superclass[layer_idx]
                        expanded_logits = sg_module[layer_idx].HKP(
                            F.log_softmax(logits_sg, dim=-1),
                            subclass_to_superclass_i,
                            len(classes),
                            logits_vg
                        )

                        expanded_logits_list.append(expanded_logits)

                    if expanded_logits_list:
                        avg_expanded_logits = torch.stack(expanded_logits_list).mean(dim=0)
                        logits_mix = args.mix_weight * F.log_softmax(logits_vg, dim=-1) + \
                                     (1 - args.mix_weight) * avg_expanded_logits
                    else:
                        logits_mix = F.log_softmax(logits_vg, dim=-1)

                    subclass_loss = F.cross_entropy(logits_mix, y_subclass, reduction='mean')
                    total_superclass_loss = sum(superclass_losses) / len(superclass_losses) if superclass_losses else 0

                    total_loss = subclass_loss + total_superclass_loss

                total_loss.backward()
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    pred_subclass = torch.argmax(logits_vg, 1)
                    pred_mix = torch.argmax(logits_mix, 1)
                    clip_model.logit_scale.data = torch.clamp(clip_model.logit_scale.data, 0, 4.6052)

                    total_num += batch_size
                    true_num += pred_subclass.eq(y_subclass).float().sum().item()
                    true_num_mix += pred_mix.eq(y_subclass).float().sum().item()
                    loss_sum += total_loss.item() * batch_size

                    for layer_idx in range(args.num_sg_layers):
                        pred_sg = torch.argmax(logits_sg, 1)
                        true_num_sg_list[layer_idx] += pred_sg.eq(y_super).float().sum().item()

            acc = true_num / total_num
            acc_sg_list = [num_sg / total_num for num_sg in true_num_sg_list]
            acc_mix = true_num_mix / total_num

            progress_info = {
                'Epoch': epoch + 1,
                'Train_Acc': f'{acc_mix:.3f}',
                'Loss': f'{loss_sum / total_num:.3f}',
            }

            for layer_idx in range(args.num_sg_layers):
                progress_info[f'Train_L{layer_idx + 1}'] = f'{acc_sg_list[layer_idx]:.3f}'

            progress_bar.set_postfix(progress_info)

            acc, acc_sg_list, acc_mix = test_model(testloader, gamma=args.mix_weight)

            test_info = {
                'Epoch': epoch + 1,
                'Test_Acc': f'{acc_mix:.3f}',
                'Best_Acc': f'{best_acc_mix:.3f}',
            }

            for layer_idx in range(args.num_sg_layers):
                test_info[f'TestL{layer_idx + 1}'] = f'{acc_sg_list[layer_idx]:.3f}'

            progress_bar.set_postfix(test_info, refresh=False)

            if acc_mix > best_acc_mix:
                best_acc_mix = acc_mix
                state_dict = {
                    "vg_module": vg_module.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc_mix,
                    "num_vg_scales": args.num_vg_scales,
                    "n_crops": args.n_crops,
                    "entropy_threshold": args.entropy_threshold,
                }

                for layer_idx in range(args.num_sg_layers):
                    state_dict[f"expert_moe_father_{layer_idx}"] = sg_module[layer_idx].state_dict()

                torch.save(state_dict, os.path.join(save_path, 'best_model.pth'))

            progress_bar.update(1)

        progress_bar.close()

        print(f"\n{'=' * 60}")
        print("Training complete")
        print(f"{'=' * 60}")
        print(f"Number of superclass layers: {args.num_sg_layers}")
        print(f"Number of visual granularity scales: {args.num_vg_scales}")
        print(f"Number of random crops: {args.n_crops}")
        print(f"Entropy threshold: {args.entropy_threshold}")
        print(f"Best accuracy: {best_acc_mix:.4f} ({best_acc_mix * 100:.2f}%)")
        print(f"Model saved path: {os.path.join(save_path, 'best_model.pth')}")
        print(f"{'=' * 60}")