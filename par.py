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
        # print('ingresso')
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_spatial = torch.cat([avg_out, max_out], dim=1)
        # print('x_spatial prima operazione', x_spatial.shape)
        x_spatial = self.spatial_attention(x_spatial)
        # print('x_spatial dopo operazione', x_spatial.shape)

        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        x_channel = avg_out + max_out
        # print('x_channel prima operazione', x_channel.shape)
        x_channel = self.sigmoid(x_channel)
        # print('x_channel dopo operazione', x_channel.shape)
        
        out = x * x_channel * x_spatial
        # print('uscita', out.shape)
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

        # Backbone ResNet-50
        self.backbone = ResNet50Backbone()

        # Moduli di attenzione spaziale e di canale per ogni attributo
        self.attention_modules = nn.ModuleList([AttentionModule(in_channels=2048) for _ in range(num_attributes)]) # Crea una lista di moduli di attenzione spaziale e di canale per ogni attributo.
        # nn.ModuleList viene utilizzato per contenere i moduli in modo che PyTorch li tracci correttamente come parte del modello.
       
        # Classificatori per ogni attributo
        # binary_classifier = [ClassifierBlock(nn.Linear(2048, 2), nn.Sigmoid()) for _ in range(3)]
        binary_classifier = [BinaryClassifier() for _ in range(3)]
      
        # multi_classifier = [ClassifierBlock(nn.Linear(2048, 11)) for _ in range(2)]
        multi_classifier = [MultiClassifier() for _ in range(2)]
     
        self.classifiers = nn.ModuleList(multi_classifier + binary_classifier) # Crea una lista di classificatori lineari per ogni attributo.

    def forward(self, x):
        # Passa l'input attraverso il backbone
        features = self.backbone(x)
        # print("dimensione di features: ", features.size())
        pred_list=[]
        attention_outputs = [attention(features) for attention in self.attention_modules]
        # print("Dimensione attention_outputs:", attention_outputs[0].size())
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