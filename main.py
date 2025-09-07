from fastapi import FastAPI, File, UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO




# Create FastAPI app
app = FastAPI()

# Allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    MODEL = tf.keras.models.load_model(
    r"C:\Users\piyus\Web Development Projects\SIH Prototype AI Integrated\hackathon_ai_model_version6.keras"
)

    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    MODEL = None

CLASS_NAMES = ['clean', 'dirty', 'potholes']


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}


def read_file_as_image(data: bytes):
    """Convert uploaded file into a processed image array."""
    try:
        # Load and convert image to RGB
        image = Image.open(BytesIO(data)).convert("RGB")
        print("Original image size:", image.size)

        # Resize to match training
        image = image.resize((256, 256))
        img_array = np.array(image)  # <-- REMOVED /255.0 here
        print("Processed image shape:", img_array.shape, "dtype:", img_array.dtype)

        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        print("Batch shape:", img_batch.shape)

        # Run prediction
        predictions = MODEL.predict(img_batch)
        print("Raw predictions:", predictions)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {"class": predicted_class, "confidence": confidence}

    except Exception as e:
        print("Error in read_file_as_image:", e)
        return {"error": str(e)}



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and return prediction."""
    try:
        contents = await file.read()
        result = read_file_as_image(contents)
        return result
    except Exception as e:
        print("Error during prediction:", e)
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




























# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from io import BytesIO

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load the trained model
# MODEL = tf.keras.models.load_model(
#     r"C:\Users\chhet\OneDrive\Desktop\Machine_Learning\hackathon_ai_model_version6.keras"
# )
# CLASS_NAMES = ['clean', 'dirty', 'potholes']

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     print("Request received")
#     contents = await file.read()
#     print(f"File size: {len(contents)} bytes")

#     image = Image.open(BytesIO(contents)).convert("RGB")
#     image = image.resize((256, 256))
#     img_array = np.array(image)
#     img_batch = np.expand_dims(img_array, axis=0)
#     print(f"Image batch shape: {img_batch.shape}")

#     predictions = MODEL.predict(img_batch)
#     print(f"Predictions: {predictions}")

#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = float(np.max(predictions[0]))

#     return {"class": predicted_class, "confidence": confidence}
