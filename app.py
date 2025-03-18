from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import re
import threading
import time

from train_model import DepthEstimationModel, train_model, setup_gpu, print_gpu_info

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to track training progress
training_in_progress = False
training_progress = 0
current_epoch = 0
total_epochs = 0
current_batch = 0
total_batches = 0
current_loss = 0.0

# Initialize GPU if available
has_gpu = setup_gpu()

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DepthEstimationModel().to(device)

# Check if model file exists and load it
model_path = 'static/model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    if torch.cuda.is_available():
        print("Model running on GPU")
        print_gpu_info()
    else:
        print("Model running on CPU")
else:
    print(f"Model file not found at {model_path}. Please train the model first.")

# Image preprocessing
def preprocess_image(image):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to the input size expected by the model
    image = image.resize((320, 240))
    
    # Transform to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device, non_blocking=True if has_gpu else False)
    return input_tensor

# Process depth map for visualization
def process_depth_map(depth_map):
    # Normalize depth map for visualization
    depth_map = depth_map.squeeze().cpu().detach().numpy()
    
    # Scale the depth map for better visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min) * 255.0
    
    # Convert to heatmap for better visualization
    depth_map = depth_map.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    return depth_colored

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        # Check if base64 image data is provided
        if 'imageData' in request.form:
            img_data = request.form['imageData']
            # Extract the base64 content after the "data:image" prefix
            img_data = re.sub('^data:image/.+;base64,', '', img_data)
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
        else:
            return jsonify({'error': 'No image provided'})
    else:
        # Get the image from the form data
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        img = Image.open(file.stream)
    
    # Preprocess image
    input_tensor = preprocess_image(img)
    
    # Make prediction with CUDA optimization when available
    with torch.no_grad():
        if has_gpu:
            torch.cuda.synchronize()  # Make sure GPU is synchronized
            
        depth_map = model(input_tensor)
        
        if has_gpu:
            torch.cuda.synchronize()  # Make sure GPU processing is complete
    
    # Process depth map for visualization
    depth_colored = process_depth_map(depth_map)
    
    # Save original and depth images
    img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'original.jpg')
    depth_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'depth.jpg')
    
    img.save(img_filename)
    cv2.imwrite(depth_filename, depth_colored)
    
    # Return image paths for display
    return jsonify({
        'original': f'/static/uploads/original.jpg',
        'depth': f'/static/uploads/depth.jpg'
    })

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/gpu_info')
def gpu_info():
    """Return GPU information to display in the UI"""
    if torch.cuda.is_available():
        info = {
            'has_gpu': True,
            'name': torch.cuda.get_device_name(0),
            'count': torch.cuda.device_count(),
            'memory_allocated': f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
            'memory_reserved': f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB",
            'cuda_version': torch.version.cuda
        }
    else:
        info = {
            'has_gpu': False,
            'message': "No GPU detected. Training will run on CPU (slower)."
        }
    return jsonify(info)

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_in_progress, training_progress, current_epoch, total_epochs, current_batch, total_batches, current_loss
    
    if training_in_progress:
        return jsonify({'message': 'Training is already in progress. Please wait.'})
    
    # Get parameters from the form
    epochs = int(request.form.get('epochs', 10))
    batch_size = int(request.form.get('batch_size', 32))
    learning_rate = float(request.form.get('learning_rate', 0.001))
    
    # Start training in a separate thread to not block the web app
    training_in_progress = True
    training_progress = 0
    current_epoch = 0
    total_epochs = epochs
    current_batch = 0
    total_batches = 0
    current_loss = 0.0
    
    def training_thread_function():
        global training_in_progress, training_progress, current_epoch, current_batch, total_batches, current_loss
        
        try:
            # Clean up GPU memory before training
            if has_gpu:
                torch.cuda.empty_cache()
                print("Starting training with clean GPU memory:")
                print_gpu_info()
            
            train_model(
                train_csv='nyu2_train.csv',
                test_csv='nyu2_test.csv',
                data_dir='nyu_data',
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_path='static/model.pth',
                update_progress_callback=update_progress
            )
        finally:
            training_in_progress = False
            training_progress = 100
            
            # Clean up GPU memory after training
            if has_gpu:
                torch.cuda.empty_cache()
                print("Training completed. Final GPU state:")
                print_gpu_info()
    
    training_thread = threading.Thread(target=training_thread_function)
    training_thread.daemon = True  # Set as daemon so it terminates when main thread exits
    training_thread.start()
    
    return jsonify({'message': 'Training started successfully'})

@app.route('/training_progress')
def get_training_progress():
    global training_in_progress, training_progress, current_epoch, total_epochs, current_batch, total_batches, current_loss
    
    if training_in_progress:
        progress = int((current_epoch / total_epochs) * 100)
    else:
        progress = training_progress
    
    gpu_info = {}
    if has_gpu:
        try:
            gpu_info = {
                'memory_used_gb': f"{torch.cuda.memory_allocated(0) / 1e9:.2f}",
                'memory_total_gb': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}",
                'utilization': "N/A"  # Actual utilization requires nvidia-smi which is not accessible through PyTorch
            }
        except Exception as e:
            gpu_info = {'error': str(e)}
    
    return jsonify({
        'is_training': training_in_progress,
        'progress': progress,
        'batch': current_batch,
        'total_batches': total_batches,
        'loss': current_loss,
        'gpu_info': gpu_info if has_gpu else None
    })

# Callback function to update training progress
def update_progress(epoch, total_epochs, batch=None, total_batches_param=None, loss=None):
    global current_epoch, current_batch, total_batches, current_loss
    current_epoch = epoch
    if batch is not None:
        current_batch = batch
    if total_batches_param is not None:
        total_batches = total_batches_param
    if loss is not None:
        current_loss = loss

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')