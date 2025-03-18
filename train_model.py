import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
import copy
import torch.cuda.amp as amp  # For mixed precision training

# Depth Estimation CNN Model
class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        # Encoder (ResNet-like)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Encoder blocks
        self.enc_block1 = self._make_encoder_block(64, 64, stride=1)
        self.enc_block2 = self._make_encoder_block(64, 128, stride=2)
        self.enc_block3 = self._make_encoder_block(128, 256, stride=2)
        self.enc_block4 = self._make_encoder_block(256, 512, stride=2)
        
        # Decoder blocks
        self.dec_block1 = self._make_decoder_block(512, 256)
        self.dec_block2 = self._make_decoder_block(256, 128)
        self.dec_block3 = self._make_decoder_block(128, 64)
        self.dec_block4 = self._make_decoder_block(64, 64)
        
        # Final layers
        self.upconv = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
    
    def _make_encoder_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        x = self.enc_block4(x)
        
        # Decoder
        x = self.dec_block1(x)
        x = self.dec_block2(x)
        x = self.dec_block3(x)
        x = self.dec_block4(x)
        
        # Final prediction
        x = self.upconv(x)
        x = self.final_relu(x)
        x = self.final_conv(x)
        
        return x

# Custom Dataset for NYU Depth V2
class NYUDepthDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Read CSV file
        self.data_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get file paths
        img_path = os.path.join(self.data_dir, self.data_frame.iloc[idx, 0])
        depth_path = os.path.join(self.data_dir, self.data_frame.iloc[idx, 1])
        
        # Load RGB image and depth map
        image = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        
        # Apply transformations
        if self.transform:
            image = self.transform['rgb'](image)
            depth = self.transform['depth'](depth)
        
        return {'image': image, 'depth': depth}

# Loss function
class BerHuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold
        
    def forward(self, pred, target):
        # Convert target to same shape as pred
        if target.size() != pred.size():
            target = F.interpolate(target, size=pred.size()[2:], mode='bilinear', align_corners=True)
        
        # Calculate absolute difference
        diff = torch.abs(pred - target)
        mask = (target > 0).detach()  # Only consider valid depth pixels
        
        # Calculate BerHu (reverse Huber) loss
        c = self.threshold * torch.max(diff).item()
        
        berhu_loss = torch.where(diff <= c, 
                            diff, 
                            (diff*diff + c*c) / (2*c))
        
        # Apply mask and return mean loss
        berhu_loss = berhu_loss * mask.float()
        return berhu_loss.sum() / (mask.sum() + 1e-8)

# GPU memory management
def print_gpu_info():
    """Print GPU info for monitoring"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

def setup_gpu():
    """Configure GPU for optimal performance"""
    if torch.cuda.is_available():
        # Set GPU to fixed memory allocation when possible
        try:
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True  # Speed up training
            torch.backends.cudnn.deterministic = False
            print("GPU optimizations applied successfully")
        except Exception as e:
            print(f"Warning: Error setting GPU optimizations: {e}")
        
        # Print GPU info
        print_gpu_info()
        return True
    else:
        print("No GPU available, using CPU")
        return False

# Training function
def train_model(train_csv, test_csv, data_dir, epochs=10, batch_size=32, learning_rate=0.001, save_path='model.pth', update_progress_callback=None):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Configure GPU
    has_gpu = setup_gpu()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # For mixed precision training
    use_amp = has_gpu and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(device='cuda') if use_amp else None
    
    # Data transforms
    data_transforms = {
        'rgb': transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'depth': transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor()
        ])
    }
    
    # Create datasets
    train_dataset = NYUDepthDataset(csv_file=train_csv, data_dir=data_dir, transform=data_transforms)
    test_dataset = NYUDepthDataset(csv_file=test_csv, data_dir=data_dir, transform=data_transforms)
    
    # Create data loaders with pinned memory for faster GPU transfer
    pin_memory = has_gpu
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2 if has_gpu else 0,  # More workers for GPU
        pin_memory=pin_memory,
        persistent_workers=True if has_gpu else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2 if has_gpu else 0,
        pin_memory=pin_memory,
        persistent_workers=True if has_gpu else False
    )
    
    # Initialize model
    model = DepthEstimationModel().to(device)
    
    # Check for multi-GPU system
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Define loss function and optimizer
    criterion = BerHuLoss().to(device)
    
    # Use different optimizers based on device
    if has_gpu:
        # Adam with higher weight decay for GPU training
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Print the learning rate manually
    print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

    
    # Training loop
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Attempt to benchmark model for better performance
    if has_gpu:
        try:
            # Warm up GPU with a few dummy batches
            dummy_input = torch.randn(2, 3, 240, 320, device=device)
            with torch.no_grad():
                for _ in range(10):  # Run several times to warm up GPU
                    _ = model(dummy_input)
            torch.cuda.synchronize()
            print("GPU warmup complete")
        except Exception as e:
            print(f"Warning: Error during GPU warmup: {e}")
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print('-' * 10)
        
        # Call the progress update callback
        if update_progress_callback:
            update_progress_callback(epoch + 1, epochs)
        
        # Print GPU memory info at start of epoch
        if has_gpu:
            print_gpu_info()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        accumulation_steps = 2  # Adjust to control memory usage (higher = lower memory usage)

        optimizer.zero_grad()  # Moved outside loop for gradient accumulation

        for i, batch in enumerate(train_loader):
            # Get batch data
            images = batch['image'].to(device, non_blocking=True)
            depths = batch['depth'].to(device, non_blocking=True)

            # Forward pass with mixed precision
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, depths)
            
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Update optimizer only every accumulation_steps batches
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            
            else:  # Normal training (without AMP)
                outputs = model(images)
                loss = criterion(outputs, depths) / accumulation_steps  # Normalize loss
                loss.backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            # Statistics
            running_loss += loss.item() * images.size(0)

            # Update batch progress
            if update_progress_callback:
                update_progress_callback(epoch + 1, epochs, i + 1, len(train_loader), loss.item())

            if i % 10 == 0:
                print(f"Batch {i}/{len(train_loader)}: Loss: {loss.item():.4f}")

        # Ensure final step update in case of non-multiple batch count
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            scaler.update()
            optimizer.zero_grad()
                
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss: {epoch_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device, non_blocking=True)
                depths = batch['depth'].to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, depths)
                
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(test_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Deep copy the model if best performance
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"New best model saved with validation loss: {best_loss:.4f}")
            
            # Save the best model
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    # Print total training time
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation loss: {best_loss:.4f}')
    
    # Final GPU memory cleanup
    if has_gpu:
        torch.cuda.empty_cache()
        print("Final GPU state:")
        print_gpu_info()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

if __name__ == "__main__":
    # Example usage
    train_model(
        train_csv='nyu2_train.csv',
        test_csv='nyu2_test.csv',
        data_dir='data',
        epochs=10,
        batch_size=16,
        learning_rate=0.001,
        save_path='static/model.pth'
    )