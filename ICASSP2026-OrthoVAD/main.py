from __future__ import print_function

import datetime
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import options
from OrthoVAD import OrthoVAD
from train import train


def build_dataset(args):
    if args.dataset_name == 'ucf-crime':
        from Dataset_ucf import dataset
    elif args.dataset_name == 'shanghaitech':
        from Dataset_sh import dataset
    else:
        raise ValueError(
            f'Unsupported dataset: {args.dataset_name}. '
            f'Please choose from: ucf-crime, shanghaitech.'
        )
    return dataset


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False

    args = options.parser.parse_args()
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(args.device)
    else:
        device = torch.device('cpu')
        print('CUDA is not available. Running on CPU.')

    now = datetime.datetime.now()
    save_path = os.path.join(
        args.model_name,
        '{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
    )

    Dataset = build_dataset(args)

    model = OrthoVAD().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt, map_location=device))
        print(f'Model loaded weights from {args.pretrained_ckpt}')

    train_dataset = Dataset(args=args, train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=1,
        shuffle=True
    )

    test_dataset = Dataset(args=args, train=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=10,
        pin_memory=True,
        num_workers=1,
        shuffle=False
    )

    if not os.path.exists(os.path.join('./ckpt', save_path)):
        os.makedirs(os.path.join('./ckpt', save_path))

    logger = False
    train(
        epochs=args.max_epoch,
        train_loader=train_loader,
        all_test_loader=[test_loader],
        args=args,
        model=model,
        optimizer=optimizer,
        logger=logger,
        device=device,
        save_path=save_path
    )
