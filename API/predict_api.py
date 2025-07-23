import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import base64
import os
from .cost_predictor import cost_predictor
from pydantic import BaseModel
import os
import requests
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from geopy.distance import geodesic
from .chatbot2 import load_faiss_index, query_gemini_rag

router = APIRouter()

GOOGLE_MAPS_API_KEY = "AIzaSyBYtp86acbQCcKuBaXUSXwUiw_5hFhMnWo"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the vector store
vector_store = load_faiss_index()

# --- CONFIGURATION ---
yolo_model_path = "best.pt"
classifier_model_path = "best_densenet_model.h5"
test_img_dir = "training"  # Directory with subfolders for each class

img_height = 224
img_width = 224
batch_size = 32

# --- Load models and class indices at startup ---
yolo_model = YOLO(yolo_model_path)
classifier_model = tf.keras.models.load_model(classifier_model_path)

# Prepare test data generator to get class indices
test_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_ds = test_data_gen.flow_from_directory(
    test_img_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
class_indices = test_ds.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

def get_damage_location(yolo_results):
    """Extract damage location from YOLO detection results, map class label to allowed locations if possible"""
    if not yolo_results[0].boxes:
        return "unknown"
    boxes = yolo_results[0].boxes
    best_box = boxes[0]
    # Try to use YOLO class label if available
    if hasattr(best_box, 'cls') and best_box.cls is not None:
        class_idx = int(best_box.cls[0].cpu().numpy())
        # Try to get class name from YOLO model
        if hasattr(yolo_model, 'names') and class_idx in yolo_model.names:
            yolo_label = yolo_model.names[class_idx].lower()
            # Map YOLO label to allowed locations
            label_map = {
                'damaged door': 'door',
                'damaged window': 'door',
                'damaged headlight': 'bumper',
                'damaged mirror': 'side_mirror',
                'dent': 'door',
                'damaged hood': 'hood',
                'damaged bumper': 'bumper',
                'damaged wind shield': 'roof',
                'roof': 'roof',
                'side mirror': 'side_mirror',
                'fender': 'fender',
            }
            for key, val in label_map.items():
                if key in yolo_label:
                    return val
    # Fallback: use heuristic based on box center
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    if center_y < 0.3:
        return "hood"
    elif center_y < 0.6:
        if center_x < 0.3:
            return "fender"
        elif center_x < 0.7:
            return "door"
        else:
            return "fender"
    else:
        return "bumper"

def get_damage_severity(yolo_results, damage_area):
    """Determine damage severity based on bounding box area (damage_area)"""
    if not yolo_results[0].boxes:
        return "minor"
    # Use area thresholds (tune as needed)
    if damage_area < 0.05:
        return "minor"
    elif damage_area < 0.15:
        return "moderate"
    else:
        return "severe"

def get_damage_area(yolo_results, img_shape):
    """Calculate normalized damage area from YOLO detection results"""
    if not yolo_results[0].boxes:
        return 0.0
    boxes = yolo_results[0].boxes
    best_box = boxes[0]
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
    box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    img_area = img_shape[0] * img_shape[1]
    return float(box_area) / float(img_area) if img_area > 0 else 0.0

class CostRequest(BaseModel):
    brand: str
    location: str = None
    severity: str = None
    car_price_lakhs: float = 15

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # YOLO detection
        yolo_results = yolo_model(img)
        result_img = yolo_results[0].plot()
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', result_img)
        yolo_result_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Classifier prediction
        img_for_classifier = cv2.resize(img, (224, 224))
        img_for_classifier = img_for_classifier / 255.0
        img_for_classifier = np.expand_dims(img_for_classifier, axis=0)
        
        classifier_prediction = classifier_model.predict(img_for_classifier)
        classifier_label = "Dent" if classifier_prediction[0][0] > 0.5 else "No Dent"
        classifier_confidence = float(classifier_prediction[0][0])

        # Extract damage location, severity, and area from detection results
        damage_location = get_damage_location(yolo_results)
        damage_area = get_damage_area(yolo_results, img.shape)
        damage_severity = get_damage_severity(yolo_results, damage_area)

        # Cost prediction using detected location, severity, and area
        cost_prediction = cost_predictor.predict_cost(
            brand="Toyota",  # Default brand, can be updated by user
            location=damage_location,
            severity=damage_severity,
            car_price_lakhs=15,  # Default price, can be updated by user
            damage_area=damage_area
        )

        return {
            "classifier_label": classifier_label,
            "classifier_confidence": classifier_confidence,
            "yolo_result_image_base64": yolo_result_image_base64,
            "detected_location": damage_location,
            "detected_severity": damage_severity,
            "damage_area": damage_area,
            "cost_prediction": cost_prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-cost")
async def predict_cost(request: CostRequest):
    try:
        location = request.location or "hood"
        severity = request.severity or "moderate"
        prediction = cost_predictor.predict_cost(
            brand=request.brand,
            location=location,
            severity=severity,
            car_price_lakhs=request.car_price_lakhs
        )
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Send a message to the chatbot and get a response.
    
    Args:
        message: The user's message
        
    Returns:
        The bot's response
    """
    try:
        if isinstance(vector_store, str):
            raise HTTPException(status_code=500, detail=vector_store)
        
        # Add formatting instructions to the user's message
        formatted_message = f"""
Please provide a response to this car repair question with the following format:

# [Main Heading]
• [First step or point]
• [Second step or point]
• [Third step or point]
etc.

Question: {message.message}
"""
        response = query_gemini_rag(formatted_message, vector_store)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a simple test endpoint
@app.get("/")
async def root():
    return {"message": "API is running. Use /predict endpoint with POST request to process images."}

@app.get("/google-places-nearby")
def google_places_nearby(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: int = Query(5000),
    sort_by: str = Query("distance")  # "distance" or "rating"
):
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={lat},{lon}&radius={radius}&type=car_repair&key={GOOGLE_MAPS_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()
    print(data)  # Debug: print the raw Google API response

    for shop in data.get("results", []):
        shop_lat = shop["geometry"]["location"]["lat"]
        shop_lon = shop["geometry"]["location"]["lng"]
        shop["distance_km"] = geodesic((lat, lon), (shop_lat, shop_lon)).km
        shop["rating"] = shop.get("rating", 0)

    shops = data.get("results", [])
    if sort_by == "rating":
        shops = sorted(shops, key=lambda x: x["rating"], reverse=True)
    else:
        shops = sorted(shops, key=lambda x: x["distance_km"])

    return JSONResponse(content={"results": shops})

@app.get("/google-place-details")
def google_place_details(place_id: str):
    url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}&fields=formatted_phone_number,opening_hours&key={GOOGLE_MAPS_API_KEY}"
    )
    response = requests.get(url)
    return JSONResponse(content=response.json())

# Add mock data for repair shops
MOCK_REPAIR_SHOPS = [
    {
        "name": "AutoCare Center",
        "vicinity": "123 Main Street, Mumbai",
        "geometry": {
            "location": {
                "lat": 19.1345,
                "lng": 72.8340
            }
        },
        "rating": 4.5,
        "opening_hours": {
            "open_now": True
        },
        "phone": "+91 9876543210"
    },
    {
        "name": "Quick Fix Garage",
        "vicinity": "456 Park Road, Mumbai",
        "geometry": {
            "location": {
                "lat": 19.1335,
                "lng": 72.8330
            }
        },
        "rating": 4.2,
        "opening_hours": {
            "open_now": False
        },
        "phone": "+91 9876543211"
    },
    {
        "name": "Pro Auto Service",
        "vicinity": "789 Lake View, Mumbai",
        "geometry": {
            "location": {
                "lat": 19.1350,
                "lng": 72.8345
            }
        },
        "rating": 4.8,
        "opening_hours": {
            "open_now": True
        },
        "phone": "+91 9876543212"
    }
]

@app.get("/server-location")
async def get_server_location():
    # For now, return a fixed location in Mumbai
    return {
        "latitude": 19.133,
        "longitude": 72.822
    }

# @app.get("/nearby-shops")
# async def get_nearby_shops(lat: float, lon: float):
#     # For now, return mock data
#     # In a real application, you would:
#     # 1. Use Google Places API or your own database
#     # 2. Filter shops based on the provided lat/lon
#     # 3. Calculate actual distances
#     return MOCK_REPAIR_SHOPS

@app.get("/python-style-nearby-shops")
def python_style_nearby_shops(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: int = Query(5000)
):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius={radius}&type=car_repair&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    print(data)  # Debug: print the raw Google API response

    shops = []
    for shop in data.get("results", []):
        # Filter out shops that are not OPERATIONAL or are permanently closed
        if shop.get("business_status") != "OPERATIONAL":
            continue
        if shop.get("permanently_closed", False):
            continue
        shop_lat = shop["geometry"]["location"]["lat"]
        shop_lon = shop["geometry"]["location"]["lng"]
        shop["distance_km"] = geodesic((lat, lon), (shop_lat, shop_lon)).km
        shop["rating"] = shop.get("rating", 0)
        shops.append(shop)

    # Sort by rating (highest first)
    shops = sorted(shops, key=lambda x: x["rating"], reverse=True)
    # Take top 5
    top_shops = shops[:5]

    # Prepare a simple response with name, rating, distance, and vicinity
    result = [
        {
            "name": shop["name"],
            "rating": shop["rating"],
            "distance_km": round(shop["distance_km"], 2),
            "vicinity": shop.get("vicinity", "")
        }
        for shop in top_shops
    ]
    return {"top_shops": result}

@app.get("/raw-google-places")
def raw_google_places(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: int = Query(5000),
    type_: str = Query("car_repair")
):
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={lat},{lon}&radius={radius}&type={type_}&key={GOOGLE_MAPS_API_KEY}"
    )
    response = requests.get(url)
    return response.json()
