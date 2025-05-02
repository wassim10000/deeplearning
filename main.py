from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "../models/1.1"
logger.info(f"Model path exists: {os.path.exists(MODEL_PATH)}")

# Global variable for model
MODEL = None
IMAGE_SIZE = (32, 32)  # Adjusted based on error (64 values = 4x4x4 or 8x8x1 or similar)
CLASS_NAMES = ["Potato___Brûlure_alternarienne", "Potato___Late_blight", "Potato___sain"]

def load_model():
    global MODEL
    # Try to load the model with better error handling
    try:
        # Set lower logging level to reduce noise
        tf.get_logger().setLevel('ERROR')
        
        # Try loading with explicit compile=False which sometimes helps with JSON errors
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully with tf.keras.models.load_model!")
        return True
    except Exception as e:
        logger.error(f"Error loading model with keras method: {str(e)}")
        logger.info("Trying alternative loading method...")
        
        try:
            # Alternative loading method
            MODEL = tf.saved_model.load(MODEL_PATH)
            logger.info("Model loaded with tf.saved_model.load successfully!")
            return True
        except Exception as e2:
            logger.error(f"Alternative loading also failed: {str(e2)}")
            logger.error("Please check if your model is correctly saved and compatible with your TensorFlow version")
            return False

# Load model at startup
if not load_model():
    logger.critical("Could not load model - exiting application")
    sys.exit(1)

@app.get("/ping")
async def ping():
    return {"status": "alive", "message": "Potato disease detection API is running"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        raise ValueError(f"Invalid image format: {str(e)}")

def preprocess_image(image):
    """Preprocess image to match model's expected input"""
    global IMAGE_SIZE  # Declare global here first
    
    # First, try to get model input shape if available
    input_shape = None
    if hasattr(MODEL, 'input_shape'):
        input_shape = MODEL.input_shape
        if input_shape is not None and len(input_shape) >= 3:
            height, width = input_shape[1], input_shape[2]
            if height is not None and width is not None:
                logger.info(f"Detected model input shape: {input_shape}")
                IMAGE_SIZE = (height, width)
    
    logger.info(f"Using image size for preprocessing: {IMAGE_SIZE}")
    
    # Resize image to expected input size
    image = tf.image.resize(image, IMAGE_SIZE)
    
    # Normalize pixel values to [0, 1] if your model expects it
    if image.dtype != np.float32:
        image = image / 255.0
    
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...), size: int = None):
    global IMAGE_SIZE  # Declare global first
    
    try:
        logger.info(f"Received prediction request for file: {file.filename}")
        
        # Read and preprocess image
        image_data = await file.read()
        original_image = read_file_as_image(image_data)
        logger.info(f"Image shape before preprocessing: {original_image.shape}")
        
        # Override image size if provided in request
        if size is not None:
            IMAGE_SIZE = (size, size)
            logger.info(f"Using provided image size: {IMAGE_SIZE}")
        
        # Get model input info for debugging
        if not hasattr(MODEL, 'predict'):
            # For SavedModel format
            infer = MODEL.signatures["serving_default"]
            # Get input tensor info
            input_name = list(infer.structured_input_signature[1].keys())[0]
            input_shape = infer.structured_input_signature[1][input_name].shape
            logger.info(f"Model expects input shape: {input_shape}")
            
            # Try to extract expected dimensions
            if len(input_shape) >= 4 and input_shape[1] is not None and input_shape[2] is not None:
                IMAGE_SIZE = (input_shape[1], input_shape[2])
                logger.info(f"Adjusted IMAGE_SIZE to: {IMAGE_SIZE}")
        
        # Preprocess image
        processed_image = preprocess_image(original_image)
        logger.info(f"Image shape after preprocessing: {processed_image.shape}")
        
        # Prepare batch dimension
        img_batch = np.expand_dims(processed_image, 0)
        
        # Make prediction based on model type
        try:
            if hasattr(MODEL, 'predict'):
                logger.info("Using model.predict() method")
                prediction = MODEL.predict(img_batch)
                logger.info(f"Raw prediction shape: {prediction.shape}")
            else:
                # For models loaded with tf.saved_model.load()
                logger.info("Using saved_model signature")
                infer = MODEL.signatures["serving_default"]
                
                # Get input tensor name
                input_name = list(infer.structured_input_signature[1].keys())[0]
                logger.info(f"Model input tensor name: {input_name}")
                
                # Try different data formats if needed
                try:
                    # First attempt with float32
                    input_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
                    inputs = {input_name: input_tensor}
                    logger.info(f"Attempting prediction with input shape: {input_tensor.shape}")
                    prediction = infer(**inputs)
                except Exception as tensor_error:
                    # If that fails, try with the exact dimensions from the model's input signature
                    logger.warning(f"First attempt failed: {str(tensor_error)}")
                    logger.info("Trying with reshaped tensor to match expected input")
                    
                    # Get expected input shape 
                    expected_shape = infer.structured_input_signature[1][input_name].shape
                    if None in expected_shape:
                        # Replace None with appropriate values
                        shape_list = list(expected_shape)
                        if shape_list[0] is None:  # Batch dimension
                            shape_list[0] = 1
                        expected_shape = tuple(shape_list)
                    
                    logger.info(f"Reshaping input to: {expected_shape}")
                    
                    # Try to reshape the input to match exactly what the model expects
                    try:
                        input_tensor = tf.reshape(input_tensor, expected_shape)
                        inputs = {input_name: input_tensor}
                        prediction = infer(**inputs)
                    except Exception as reshape_error:
                        logger.error(f"Reshape failed: {str(reshape_error)}")
                        # One more attempt with a differently sized image
                        try:
                            # Try with 16x16
                            new_size = (16, 16)
                            logger.info(f"Attempting with image size: {new_size}")
                            resized = tf.image.resize(original_image, new_size)
                            resized = resized / 255.0  # Normalize
                            input_tensor = tf.expand_dims(resized, 0)
                            inputs = {input_name: input_tensor}
                            prediction = infer(**inputs)
                        except Exception as size_error:
                            logger.error(f"16x16 size failed: {str(size_error)}")
                            # Finally try with grayscale
                            try:
                                resized = tf.image.resize(original_image, (32, 32))
                                # Convert to grayscale (1 channel)
                                grayscale = tf.image.rgb_to_grayscale(resized)
                                grayscale = grayscale / 255.0
                                input_tensor = tf.expand_dims(grayscale, 0)
                                inputs = {input_name: input_tensor}
                                prediction = infer(**inputs)
                                logger.info("Grayscale transformation worked!")
                            except Exception as gray_error:
                                logger.error(f"Grayscale attempt failed: {str(gray_error)}")
                                raise HTTPException(
                                    status_code=500,
                                    detail="All input format attempts failed. Model input requirements might be different."
                                )
                
                # Extract the output tensor
                output_name = list(prediction.keys())[0]
                logger.info(f"Model output tensor name: {output_name}")
                prediction = prediction[output_name].numpy()
                logger.info(f"Raw prediction shape: {prediction.shape}")
            
            # Process prediction results
            if len(prediction.shape) == 2:
                # Standard classification output [batch_size, num_classes]
                predicted_class_idx = np.argmax(prediction[0])
                confidence = float(np.max(prediction[0]))
            else:
                # Handle other output formats if needed
                logger.warning(f"Unexpected prediction shape: {prediction.shape}")
                prediction_flat = prediction.flatten()
                predicted_class_idx = np.argmax(prediction_flat)
                confidence = float(np.max(prediction_flat))
            
            if predicted_class_idx < len(CLASS_NAMES):
                predicted_class = CLASS_NAMES[predicted_class_idx]
            else:
                logger.error(f"Predicted index {predicted_class_idx} is out of bounds for CLASS_NAMES")
                predicted_class = "Unknown"
            
            logger.info(f"Prediction: {predicted_class} with confidence {confidence:.4f}")
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'class_probabilities': {
                    class_name: float(prediction[0][i]) 
                    for i, class_name in enumerate(CLASS_NAMES) 
                    if i < len(prediction[0])
                }
            }
            
        except Exception as pred_error:
            logger.error(f"Error during prediction: {str(pred_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Prediction processing error: {str(pred_error)}"
            )
            
    except ValueError as ve:
        logger.error(f"Value error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Endpoint to get model information for debugging"""
    if MODEL is None:
        return {"status": "error", "message": "Model not loaded"}
    
    # Get model information
    model_info = {
        "model_loaded": MODEL is not None,
        "model_type": str(type(MODEL))
    }
    
    # Add additional info for Keras models
    if hasattr(MODEL, 'input_shape'):
        model_info["input_shape"] = str(MODEL.input_shape)
    if hasattr(MODEL, 'output_shape'):
        model_info["output_shape"] = str(MODEL.output_shape)
    
    # For SavedModel format
    if hasattr(MODEL, 'signatures'):
        signature_keys = list(MODEL.signatures.keys())
        model_info["signatures"] = signature_keys
        
        if "serving_default" in signature_keys:
            serving = MODEL.signatures["serving_default"]
            model_info["input_signature"] = str(serving.structured_input_signature)
            
            # Get detailed input shape information
            try:
                input_name = list(serving.structured_input_signature[1].keys())[0]
                input_tensor_spec = serving.structured_input_signature[1][input_name]
                model_info["input_name"] = input_name
                model_info["input_shape"] = str(input_tensor_spec.shape)
                model_info["input_dtype"] = str(input_tensor_spec.dtype)
                
                # Calculate total elements expected
                shape = input_tensor_spec.shape
                if None not in shape:
                    total_elements = 1
                    for dim in shape:
                        total_elements *= dim
                    model_info["total_input_elements"] = total_elements
                else:
                    # Calculate with batch size of 1 if batch dimension is None
                    shape_list = list(shape)
                    if shape_list[0] is None:
                        shape_list[0] = 1
                    total_elements = 1
                    for dim in shape_list:
                        if dim is not None:
                            total_elements *= dim
                    model_info["total_input_elements"] = f"{total_elements} (with batch size 1)"
                
                # Get output shape information
                output_name = list(serving.structured_outputs.keys())[0]
                output_tensor_spec = serving.structured_outputs[output_name]
                model_info["output_name"] = output_name
                model_info["output_shape"] = str(output_tensor_spec.shape)
                model_info["output_dtype"] = str(output_tensor_spec.dtype)
                
            except Exception as e:
                model_info["shape_error"] = str(e)
                
    # Add suggested image size based on model input
    model_info["suggested_image_size"] = str(IMAGE_SIZE)
            
    return model_info

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)