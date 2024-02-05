import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np

SHOW_CROP = False
WIDTH_PAR = 224
HEIGHT_PAR = 224

class ResNet50Backbone(nn.Module):

    def __init__(self):
        super(ResNet50Backbone, self).__init__() # Chiama il costruttore della classe madre nn.Module per inizializzare la classe base.

        self.model = resnet50(pretrained=True) # Carica il modello preaddestrato ResNet-50 dal torchvision.models con i pesi preaddestrati su ImageNet.
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        return self.model(x)

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

class AttentionModule(nn.Module):

    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()

        self.channel_attention  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_attention(x)
        # print("channel_attention: ", channel_att.size())
        spatial_att = self.spatial_attention(x)
        # print("spatial_attention: ", spatial_att.size())
        x = x * channel_att * spatial_att
        return x

class ClassifierBlock(nn.Module):

  def __init__(self, layer, activation_function = None):
    super(ClassifierBlock, self).__init__()

    self.block = nn.Sequential(layer, activation_function) if activation_function else nn.Sequential(layer)

  def forward(self, x):
    return self.block(x)

class AttributeRecognitionModel(nn.Module):

    def __init__(self, num_attributes):
        super(AttributeRecognitionModel, self).__init__()

        # Backbone ResNet-50
        self.backbone = ResNet50Backbone()

        # Moduli di attenzione spaziale e di canale per ogni attributo
        self.attention_modules = nn.ModuleList([AttentionModule(in_channels=2048) for _ in range(num_attributes)]) # Crea una lista di moduli di attenzione spaziale e di canale per ogni attributo.
        # nn.ModuleList viene utilizzato per contenere i moduli in modo che PyTorch li tracci correttamente come parte del modello.

        # Classificatori per ogni attributo
        binary_classifier = [ClassifierBlock(nn.Linear(2048, 2), nn.Sigmoid()) for _ in range(3)]

        multi_classifier = [ClassifierBlock(nn.Linear(2048, 11)) for _ in range(2)]

        self.classifiers = nn.ModuleList(multi_classifier + binary_classifier) # Crea una lista di classificatori lineari per ogni attributo.

    def forward(self, x):
        # Passa l'input attraverso il backbone
        features = self.backbone(x)
        # print("dimensione di features: ", features.size())
        pred_list=[]
        attention_outputs = [attention(features) for attention in self.attention_modules]

        for att_output, classifier in zip(attention_outputs, self.classifiers):
            # print("Dimensione att_output:", att_output.size())
            flattened_output = att_output.view(att_output.size(0), -1)
            # print("Dimensione flattened_output:", flattened_output.size())
            pred = classifier(flattened_output)
            # print("Dimensione logits:", pred.size())
            pred_list.append(pred)
        # print("Dimensione logits totale:", len(pred_list))
        # print('--- pred_list ', pred_list)
        return pred_list

    def freeze_backbone_parameters(self):
      self.backbone.freeze_all()

    def unfreeze_parameters(self):
        for param in self.attention_modules.parameters():
            param.requires_grad = True

        for param in self.classifiers.parameters():
            param.requires_grad = True
