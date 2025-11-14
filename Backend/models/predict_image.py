import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load("models/parkinson_image_model.pth", map_location=torch.device('cpu'))
model.eval()

# Define same transform used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to predict image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    
    classes = ['healthy', 'parkinson']
    return classes[predicted.item()]

# Example usage
if __name__ == "__main__":
    result = predict_image("datasets/images/test/healthy/healthy_test1.jpg")
    print(f"Prediction: {result}")
