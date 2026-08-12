"""UTRNet recognition model (adapted from webapp model.py + modules/)."""

import numpy as np
import torch
import torch.nn as nn

from ingestion.urdu_ocr.unet import UNet


class UNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super().__init__()
        self.ConvNet = UNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)


class DropoutLayer(nn.Module):
    """Feature-wise dropout mask (random per timestep, upstream keeps it at inference)."""

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input):
        nums = (np.random.rand(input.shape[1]) > 0.2).astype(int)
        mask = torch.from_numpy(nums).to(self.device)
        mask = torch.reshape(mask, (input.shape[1], 1)).to(self.device)
        mask = mask.repeat(input.shape[0], 1, input.shape[2]).to(self.device)
        return input * mask


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        return self.linear(recurrent)


class Model(nn.Module):
    """UNet extractor + temporal dropout + 2x BiLSTM + linear prediction."""

    def __init__(self, num_class=181, device="cpu"):
        super().__init__()
        self.device = device
        self.FeatureExtraction = UNet_FeatureExtractor(1, 512)
        self.FeatureExtraction_output = 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.dropout1 = DropoutLayer(self.device)
        self.dropout2 = DropoutLayer(self.device)
        self.dropout3 = DropoutLayer(self.device)
        self.dropout4 = DropoutLayer(self.device)
        self.dropout5 = DropoutLayer(self.device)
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256),
        )
        self.SequenceModeling_output = 256
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input, text=None, is_train=True):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)
        branches = [
            self.SequenceModeling(dropout(visual_feature))
            for dropout in (self.dropout1, self.dropout2, self.dropout3, self.dropout4, self.dropout5)
        ]
        contextual_feature = branches[0]
        for branch in branches[1:]:
            contextual_feature = contextual_feature.add(branch)
        contextual_feature = contextual_feature * (1 / 5)
        return self.Prediction(contextual_feature.contiguous())
