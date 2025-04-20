# Bone Fracture Detector

**Train** a ResNet50-based model to classify X-ray images into `fracture` or `normal`.

## Setup

1. Place your dataset into `data/` with subfolders:
   ```
   data/
     fracture/
       img1.png
       ...
     normal/
       img2.png
       ...
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train the model:
   ```
   python train.py
   ```
4. Run inference:
   ```
   python infer.py path/to/new_image.png
   ```

## Files

- `train.py`: Training script.  
- `infer.py`: Inference script.  
- `requirements.txt`: Dependencies.  
- `bone_fracture_detector.h5`: Generated model after training.  
- `data/`: Empty folder for your images.  
