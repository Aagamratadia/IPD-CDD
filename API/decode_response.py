import requests
import base64
from PIL import Image
import io
import json

def get_prediction(image_path):
    # Send request to API
    url = "http://127.0.0.1:8000/predict"
    files = {"file": open(image_path, "rb")}
    
    # Make the POST request
    response = requests.post(url, files=files)
    
    # Parse the JSON response
    result = response.json()
    
    # Print classification results
    print(f"Predicted class: {result['classifier_label']}")
    print(f"Confidence: {result['classifier_confidence']:.4f}")
    
    # Decode and save the YOLO detection image
    try:
        # Decode base64 string to image
        image_data = base64.b64decode(result["yolo_result_image_base64"])
        image = Image.open(io.BytesIO(image_data))
        
        # Save the image
        output_path = "detection_result.jpg"
        image.save(output_path)
        print(f"Detection image saved as: {output_path}")
        
        # Optionally display the image
        image.show()
        
    except Exception as e:
        print(f"Error processing image: {e}")

# Use the function
image_path = "test_images/test/1.jpg"  # Replace with your image path
get_prediction(image_path)
