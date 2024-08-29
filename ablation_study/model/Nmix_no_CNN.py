import torch
import torch.nn as nn
import torch.nn.functional as F

class RNASeqClassifier(nn.Module):
    def __init__(self, num_layers=1, num_heads=1, drop_out=0.2, in_channels1=4, in_channels2=3):
        super(RNASeqClassifier, self).__init__()
        
        
        # Initialize multi-head attention layers
        self.attention1 = nn.MultiheadAttention(embed_dim=in_channels1, num_heads=num_heads)
        self.attention2 = nn.MultiheadAttention(embed_dim=in_channels2, num_heads=num_heads)
        self.attention3 = nn.MultiheadAttention(embed_dim=41, num_heads=num_heads)

        # Initialize fully connected layers
        self.fc_layers = self._make_fc_layers(in_channels1 + in_channels2 +41, [32, 16, 8, 4, 2], drop_out)

    def forward(self, x1, x2, x3):
        # Rearrange dimensions for multi-head attention layer input format
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        x3 = x3.permute(0, 2, 1)

        identity1, identity2, identity3 = x1, x2, x3

        # Pass x1, x2, x3 through attention layers
        x1, _ = self.attention1(x1, x1, x1)
        x2, _ = self.attention2(x2, x2, x2)
        x3, _ = self.attention3(x3, x3, x3)

        # Residual connection
        x1 += identity1
        x2 += identity2
        x3 += identity3

        # Average pooling
        x1 = torch.mean(x1, dim=1)
        x2 = torch.mean(x2, dim=1)
        x3 = torch.mean(x3, dim=1)

        # Concatenate x1, x2, x3
        x = torch.cat((x1, x2, x3), dim=1)

        # Apply fully connected layers
        x = self._apply_fc_layers(self.fc_layers, x)

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
