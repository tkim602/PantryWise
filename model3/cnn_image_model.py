import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class CNNImageModel(nn.Module):
    def __init__(self, base_model='resnet50'):
        super().__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model = nn.Sequential(*list(self.model.children())[:-1])  
        self.embedding_dim = 2048

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

def get_image_embedding(model, img_path):
    img = preprocess_image(img_path)
    with torch.no_grad():
        return model(img).numpy()
    
def load_pretrained_cnn():
    """
    Returns an instance of the CNN model with pretrained weights.
    """
    model = CNNImageModel()  # uses ResNet50 by default
    model.eval()             # set to evaluation mode
    return model