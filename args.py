import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of MLDL project feature classifier with temporal aggregation")

parser.add_argument('--source', type=str, help='path for source features')

parser.add_argument('--train_labels', type=str, help='path for train labels')

parser.add_argument('--target_features', type=str, help='path for the target features')

parser.add_argument('--val_labels', type=str, help='path for validation labels')

parser.add_argument('--aggregation', type=str, choices=['avgpool', 'trn'],
                    default='avgpool', help="list of temporal aggregations")

parser.add_argument('--modality', type=str, choices=['RGB', 'Flow'],
                    default='RGB', help="list of modalities")

parser.add_argument('--backbone', type=str, choices=['i3d', 'tsm'],
                    default='i3d', help="list backbones from which features have been extracted")