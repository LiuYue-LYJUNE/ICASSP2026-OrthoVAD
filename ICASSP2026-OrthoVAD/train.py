import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

from eval import eval_p
from test import test

binary_CE_loss = torch.nn.BCELoss(reduction='mean')


def KMXMILL_individual(element_logits, seq_len, labels, device, args):
    k = np.ceil(seq_len / args.k).astype('int32')
    instance_logits = torch.zeros(0).to(device)
    real_label = torch.zeros(0).to(device)
    real_size = int(element_logits.shape[0])

    for i in range(real_size):
        if seq_len[i] == 0:
            continue

        k[i] = min(k[i], seq_len[i])
        if k[i] == 0:
            k[i] = 1

        tmp, _ = torch.topk(element_logits[i][:seq_len[i]].squeeze(), k=int(k[i]), dim=0)
        instance_logits = torch.cat((instance_logits, tmp))

        target = torch.ones(int(k[i])) if labels[i] == 1 else torch.zeros(int(k[i]))
        real_label = torch.cat((real_label, target.to(device)))

    if instance_logits.nelement() == 0:
        return torch.tensor(0.0, device=device)

    milloss = binary_CE_loss(instance_logits, real_label)
    return milloss


def orthogonality_loss(anomaly_features, normal_features, device):
    if anomaly_features.shape[0] == 0 or normal_features.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    anomaly_video_prototypes = torch.mean(anomaly_features, dim=1)
    normal_video_prototypes = torch.mean(normal_features, dim=1)

    mean_anomaly_prototype = torch.mean(anomaly_video_prototypes, dim=0)
    mean_normal_prototype = torch.mean(normal_video_prototypes, dim=0)

    mean_anomaly_prototype = F.normalize(mean_anomaly_prototype, p=2, dim=0)
    mean_normal_prototype = F.normalize(mean_normal_prototype, p=2, dim=0)

    loss = torch.dot(mean_anomaly_prototype, mean_normal_prototype).pow(2)
    return loss


def gate_sparsity_loss(gate_values, device):
    return torch.mean(torch.abs(gate_values))


def train(epochs, train_loader, all_test_loader, args, model, optimizer, logger, device, save_path):
    [test_loader] = all_test_loader
    itr = 0

    result_dir = os.path.join('./result', save_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(os.path.join(result_dir, 'result.txt'), mode='w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))

    if args.pretrained_ckpt:
        model.load_state_dict(torch.load(args.pretrained_ckpt, map_location=device))
        print(f'Model loaded weights from {args.pretrained_ckpt}')
    else:
        print('Model is trained from scratch')

    total_iterations = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=1e-7)

    for epoch in range(epochs):
        model.train()

        for _, data in enumerate(train_loader):
            itr += 1
            [anomaly_features_v, normaly_features_v], [anomaly_label, normaly_label] = data

            visual_features = torch.cat(
                (anomaly_features_v.squeeze(0), normaly_features_v.squeeze(0)),
                dim=0
            )
            videolabels = torch.cat(
                (anomaly_label.squeeze(0), normaly_label.squeeze(0)),
                dim=0
            )

            seq_len = torch.sum(torch.max(visual_features.abs(), dim=2)[0] > 0, dim=1)
            if torch.max(seq_len) == 0:
                continue

            visual_features = visual_features[:, :torch.max(seq_len), :].float().to(device)
            videolabels = videolabels.to(device)

            element_logits, refined_features, _, gate_values = model(visual_features, seq_len)

            is_anomaly = (videolabels == 1).squeeze()
            is_normal = (videolabels == 0).squeeze()

            anomaly_refined_features = refined_features[is_anomaly]
            normal_refined_features = refined_features[is_normal]

            mil_loss = KMXMILL_individual(
                element_logits,
                seq_len.cpu().numpy(),
                videolabels.cpu().numpy(),
                device,
                args
            )
            ortho_loss = orthogonality_loss(anomaly_refined_features, normal_refined_features, device)
            gate_loss = gate_sparsity_loss(gate_values, device)

            total_loss = (
                args.lambda_mil * mil_loss
                + args.lambda_ortho * ortho_loss
                + args.lambda_gate * gate_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if itr % args.snapshot == 0 and itr != 0:
                model.eval()
                test_result_dict = test(test_loader, model, device, args)
                eval_p(
                    itr=itr,
                    dataset=args.dataset_name,
                    predict_dict=test_result_dict,
                    logger=logger,
                    save_path=save_path,
                    plot=args.plot,
                    args=args
                )
                model.train()
