import argparse

parser = argparse.ArgumentParser(description='OrthoVAD')

parser.add_argument('--device', type=int, default=0, help='GPU ID')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--model_name', default='OrthoVAD', help='name of the model/framework')
parser.add_argument('--pretrained_ckpt', default=None, help='path to pretrained checkpoint')

# Dataset and feature paths
parser.add_argument('--dataset_name', type=str, default='ucf-crime',
                    choices=['ucf-crime', 'shanghaitech'],
                    help='dataset name')
parser.add_argument('--dataset_path', type=str, default='./dataset',
                    help='path to the directory containing anomaly dataset annotations')
parser.add_argument('--feature_dir_sh', type=str, default='./features/ShanghaiTech',
                    help='path to ShanghaiTech CLIP features')
parser.add_argument('--feature_dir_ucf', type=str, default='./features/UCF-Crime',
                    help='path to UCF-Crime CLIP features')

# Training hyperparameters
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for DataLoader')
parser.add_argument('--sample_size', type=int, default=30,
                    help='number of anomalous and normal videos sampled in one training iteration')
parser.add_argument('--sample_step', type=int, default=1, help='temporal sampling step')
parser.add_argument('--max-seqlen', type=int, default=300, help='maximum sequence length during training')
parser.add_argument('--max_epoch', type=int, default=100, help='maximum number of training epochs')
parser.add_argument('--snapshot', type=int, default=80, help='evaluation interval in training iterations')

# Loss hyperparameters
parser.add_argument('--k', type=int, default=8, help='top-k divisor used in MIL loss')
parser.add_argument('--lambda_mil', type=float, default=1.0, help='weight of MIL loss')
parser.add_argument('--lambda_ortho', type=float, default=0.02, help='weight of prototype orthogonality loss')
parser.add_argument('--lambda_gate', type=float, default=1.0, help='weight of gate sparsity loss')

# Evaluation and visualization
parser.add_argument('--plot', type=int, default=0, help='whether to plot video anomaly maps during evaluation')
