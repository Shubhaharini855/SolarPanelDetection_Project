import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

def load_resnet50(checkpoint_path, device='cpu'):
    model = torch.hub.load('pytorch/vision:v0.15.2', 'resnet50', pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # binary: [no, yes]
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval().to(device)
    return model

_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def infer_image(model, image_path, device='cpu'):
    img = Image.open(image_path).convert('RGB')
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # probs[1] is probability of "solar present" assuming label order [no,yes]
    return float(probs[1])
