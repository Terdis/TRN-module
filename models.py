import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class AvgPoolClassifier(nn.Module):
    def __init__(self, img_features_dim, num_class):
        super(AvgPoolClassifier, self).__init__()
        self.img_features_dim=img_features_dim
        self.num_class=num_class

        # (input_shape - pool_size + 1)/stride
        self.avg_pool=nn.AvgPool1d(kernel_size=8, stride=1, padding=0)
        self.bn=nn.BatchNorm1d(5, eps=0.001, momentum=0.01)
        self.flatten=nn.Flatten()
        self.dropout=nn.Dropout(p=0.4)
        self.dense=nn.Linear(5*1017, 8)

    def forward(self, x):
        x=self.avg_pool(x)
        x=self.bn(x)
        x=self.flatten(x)
        x=self.dropout(x)
        x=self.dense(x)

        return x

class RelationModuleMultiScaleWithClassifier(torch.nn.Module):
    # relation module in multi-scale with a classifier at the end
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScaleWithClassifier, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] #

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        self.classifier_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]

            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),# this is the newly added thing
                        nn.Linear(num_bottleneck, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),
                        )
            classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            self.classifier_scales += [classifier]
        # maybe we put another fc layer after the summed up results???
        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        act_all = self.classifier_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = self.classifier_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))