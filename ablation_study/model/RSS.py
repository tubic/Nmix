import torch
import torch.nn as nn
import torch.nn.functional as F


class RNASeqClassifier(nn.Module):
    def __init__(self, num_layers=2, num_heads=4, drop_out=0.2, in_channels1=4, in_channels2=3):
        super(RNASeqClassifier, self).__init__()

        # Initialize 2D convolution layers for processing x3
        self.conv2d1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1)
        self.bn2d1 = nn.BatchNorm2d(8)
        self.prelu2d1 = nn.PReLU()
        self.conv2d2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1)
        self.bn2d2 = nn.BatchNorm2d(16)
        self.prelu2d2 = nn.PReLU()

        # Initialize multi-head attention layers
        self.attention2 = nn.MultiheadAttention(embed_dim=16, num_heads=num_heads)

        # Initialize fully connected layers
        self.fc_layers = self._make_fc_layers(16, [8, 4, 2], drop_out)

    def forward(self, x1, x2, x3):
        x3 = self.prelu2d1(self.bn2d1(self.conv2d1(x3.unsqueeze(1))))
        x3 = self.prelu2d2(self.bn2d2(self.conv2d2(x3))) 
        x3 = x3.view(x3.size(0), x3.size(1), -1)  
        x3 = x3.permute(0, 2, 1)
        identity3 = x3
        x3, _ = self.attention2(x3, x3, x3)
        x3 += identity3
        x3 = torch.mean(x3, dim=1)
        x = x3
        x = self._apply_fc_layers(self.fc_layers, x)

        return x

    def _apply_conv_layers(self, layers, x):
        """Apply a sequence of convolution layers to the input x"""
        for layer in layers:
            x = layer(x)
        return x

    def _make_fc_layers(self, in_features, layer_sizes, dropout_rate):
        """Create a sequence of fully connected layers"""
        layers = nn.ModuleList()
        for size in layer_sizes:
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = size
        layers.append(nn.Linear(in_features, 1))  # Final output layer
        return layers

    def _apply_fc_layers(self, layers, x):
        """Apply a sequence of fully connected layers to the input x"""
        for layer in layers:
            x = layer(x)
        return x



class Loss(nn.Module):
    def __init__(self, gamma=2, pos_weight=1, reduction='mean'):
        super(Loss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Binary cross-entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')

        # Focal loss computation
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt).pow(self.gamma)
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
