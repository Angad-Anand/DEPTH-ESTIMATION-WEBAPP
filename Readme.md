# Depth Estimation Web Application

This project is a Flask-based web application that uses deep learning to estimate depth from RGB images using the NYU Depth V2 dataset.

## Project Structure

```
DEPTH_ESTIMATION_WEBAPP/
├── data/
│   ├── nyu2_train/
│   │   └── (train image directories)
│   └── nyu2_test/
│       └── (test image directories)
├── static/
│   └── uploads/
├── templates/
│   ├── index.html
│   └── train.html
├── nyu2_train.csv
├── nyu2_test.csv
├── app.py
└── train_model.py
```

## Requirements

- Python 3.7+
- PyTorch
- Flask
- OpenCV
- Pandas
- Numpy
- Pillow

You can install the required packages using:

```bash
pip install torch torchvision flask opencv-python pandas numpy pillow
```

## Dataset

The application uses the NYU Depth V2 dataset, which needs to be organized in the following structure:

- Each training sample contains RGB images (JPG) and corresponding depth maps (PNG)
- The mapping between RGB images and depth maps is defined in CSV files (`nyu2_train.csv` and `nyu2_test.csv`)
- Each CSV row contains paths to the RGB image and its corresponding depth map

## Usage

### Running the Application

1. Clone the repository
2. Set up the dataset as described above
3. Run the Flask application:

```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

### Training the Model

1. Navigate to the "Train Model" page
2. Set the desired parameters (epochs, batch size, learning rate)
3. Click "Start Training"
4. Training will proceed in the background, with progress shown in the terminal
5. The best model will be saved to `static/model.pth`

### Using the Model

1. On the main page, you can:
   - Start your device's camera
   - Capture an image for depth estimation
   - Upload an existing image

2. The application will process the image and display:
   - The original RGB image
   - The estimated depth map (color-coded)

## Model Architecture

The model uses an encoder-decoder architecture inspired by U-Net:

- Encoder: ResNet-like blocks that progressively reduce spatial dimensions while increasing feature channels
- Decoder: Transpose convolution blocks that progressively increase spatial dimensions and decrease feature channels
- Skip connections are used to preserve spatial details

## Loss Function

The model uses the BerHu (reverse Huber) loss function, which is effective for depth estimation:
- For small residuals, it behaves like L1 loss (absolute error)
- For large residuals, it behaves like L2 loss (squared error)

## Mobile Access

To access the application from a mobile device on the same network:
1. Run the app with `app.run(debug=True, host='0.0.0.0')`
2. Find your computer's local IP address
3. Access the application from your mobile device at `http://YOUR_LOCAL_IP:5000`