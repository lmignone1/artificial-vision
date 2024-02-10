import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np

SHOW_CROP = False
WIDTH_PAR = 224
HEIGHT_PAR = 224

class ResNet50Backbone(nn.Module):

    def __init__(self):
        super(ResNet50Backbone, self).__init__()

        self.model = resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        return self.model(x)

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

class AttentionModule(nn.Module):
    #https://github.com/luuuyi/CBAM.PyTorch

    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_attention  = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_spatial = torch.cat([avg_out, max_out], dim=1)
        x_spatial = self.spatial_attention(x_spatial)

        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        x_channel = avg_out + max_out
        x_channel = self.sigmoid(x_channel)
        
        out = x * x_channel * x_spatial
        return out

class BinaryClassifier(nn.Module):

  def __init__(self):
    super(BinaryClassifier, self).__init__()

    self.block1 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3))
    self.block2 = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
  
  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    return x
  

class MultiClassifier(nn.Module):

  def __init__(self):
      super(MultiClassifier, self).__init__()

      self.block1 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3))
      self.block2 = nn.Sequential(nn.Linear(512, 11))

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    return x


class AttributeRecognitionModel(nn.Module):

    def __init__(self, num_attributes):
        super(AttributeRecognitionModel, self).__init__()

        self.backbone = ResNet50Backbone()
        self.attention_modules = nn.ModuleList([AttentionModule(in_channels=2048) for _ in range(num_attributes)])
        binary_classifier = [BinaryClassifier() for _ in range(3)]
        multi_classifier = [MultiClassifier() for _ in range(2)]
        self.classifiers = nn.ModuleList(multi_classifier + binary_classifier)

    def forward(self, x):
        features = self.backbone(x)
        pred_list=[]
        attention_outputs = [attention(features) for attention in self.attention_modules]
        
        for att_output, classifier in zip(attention_outputs, self.classifiers):
            flattened_output = att_output.view(att_output.size(0), -1)
            pred = classifier(flattened_output)
            pred_list.append(pred)

        return pred_list

    def freeze_backbone_parameters(self):
      self.backbone.freeze_all()

    def unfreeze_parameters(self):
        for param in self.attention_modules.parameters():
            param.requires_grad = True

        for param in self.classifiers.parameters():
            param.requires_grad = True