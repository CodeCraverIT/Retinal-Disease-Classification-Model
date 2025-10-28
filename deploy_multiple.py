import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ============================
# CONFIGURATION
# ============================
model_path = "resnet_retinal_multilabel.pth"
num_classes = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']

# ============================
# MODEL DEFINITION
# ============================
class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelResNet, self).__init__()
        self.base = models.resnet18(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

# ============================
# LOAD MODEL
# ============================
model = MultiLabelResNet(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ============================
# TRANSFORM
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================
# PREDICTION FUNCTION
# ============================
def predict_image_multilabel(img_path, threshold=0.5):
    # Load and preprocess image
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    # Apply threshold
    predicted = []
    for i, p in enumerate(probs):
        if p > threshold:
            predicted.append((class_names[i], float(p) * 100))

    return predicted, probs