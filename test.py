import torch
from PIL import Image


# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# Function to count people in an image
def count_people(image_path):
    # Load image with PIL
    img = Image.open(image_path)


    # Convert image to RGB (in case it's not)
    img = img.convert("RGB")
   
    # Perform inference
    results = model(img)


    # Extract results for 'person' class (class index 0 in COCO)
    people_count = (results.xyxy[0][:, -1] == 0).sum().item()
   
    return people_count


# Example usage
image_path = r'C:\Users\HP\OneDrive\Desktop\finalyear\yolov5\testing\t3.png'  # Provide the path to your image file
print("Number of people in the image:", count_people(image_path))
