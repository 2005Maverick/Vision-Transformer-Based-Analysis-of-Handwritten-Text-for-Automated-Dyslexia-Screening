import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class DyslexiaViT(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DyslexiaViT, self).__init__()
        
        # Load pretrained ViT model
        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                num_classes=num_classes
            )
            self.vit = ViTModel(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, pixel_values):
        # Get ViT outputs
        outputs = self.vit(pixel_values=pixel_values)
        
        # Get the [CLS] token output
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits

    def get_attention_weights(self, pixel_values):
        """Get attention weights for visualization"""
        outputs = self.vit(pixel_values=pixel_values, output_attentions=True)
        return outputs.attentions 