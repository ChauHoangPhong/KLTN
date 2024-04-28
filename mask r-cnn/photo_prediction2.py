import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from PIL import Image, ImageTk
import skimage
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import os

# Create TkinterDnD window
root = TkinterDnD.Tk()
root.title("Object detection in images using MASK-RCNN")
root.resizable(0, 0)

title_text = "Object detection in images using MASK-RCNN"
TITLE_LABEL = tk.Label(root, text=title_text.upper(), font=("Helvetica", 16, "bold"))
TITLE_LABEL.pack(side=tk.TOP, pady=10)

# Define the prediction configuration
class PredictionConfig(Config):
    # Define the name of the configuration
    NAME = "foods_cfg"
    # Number of classes (background + 32)
    NUM_CLASSES = 1 + 32
    # Simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# Load model
DEFAULT_LOGS_DIR = "logs_model"
cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir=DEFAULT_LOGS_DIR, config=cfg)
model.load_weights(os.path.join(DEFAULT_LOGS_DIR, 'mask_rcnn_foods_cfg_0019.h5'), by_name=True)

def apply_mask(image, mask, color, alpha=0.5):
    """Apply mask to image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                   image[:, :, c] *
                                   (1 - alpha) + alpha * color[c] * 255,
                                   image[:, :, c])
    return image


def detect_objects(image_path):
    # Load image
    foods_img = skimage.io.imread(image_path)
    detected = model.detect([foods_img])[0]
    
    # Clear previous detections
    ax.clear()
    ax.imshow(foods_img)
    
    # Loop through each detected object
    class_names = ['apple', 'banana', 'bean', 'bitter_gourd', 'bottle_gourd', 'bread', 'brinjal', 'broccoli', 'cabbage',
              'capsicum', 'carrot', 'cauliflower', 'chicken', 'coca-cola', 'cucumber', 'Egg', 'eggplant',
              'fried chicken', 'horse meat', 'meat', 'mushroom', 'orange', 'papaya', 'pepsi', 'pork', 'potato',
              'pumpkin', 'radish', 'rice', 'stir-fried kale', 'fried tofu', 'tomato']
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    
    masked_image = foods_img.copy()  # Khởi tạo hình ảnh đã áp dụng mask

    for i, box in enumerate(detected['rois']):
        class_id = detected['class_ids'][i]
        class_name = class_names[class_id - 1]
        y1, x1, y2, x2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Draw rectangle around the object
        rect = Rectangle((x1, y1), width, height, fill=False, edgecolor=colors[i % len(colors)], linewidth=2)
        ax.add_patch(rect)

        # Add label of the class to the rectangle
        ax.text(x1, y1, class_name, color='r', verticalalignment='top', bbox={'color': 'white', 'alpha': 0.7, 'pad': 2})
        
        # Visualize masks
        object_mask = detected['masks'][:, :, i]
        # Áp dụng mask riêng lẻ lên hình ảnh
        masked_image = apply_mask(masked_image, object_mask, colors[i % len(colors)], alpha=0.5)
        
    ax.imshow(masked_image.astype(np.uint8))
    canvas.draw()

def handle_upload():
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_objects(file_path)

# Create Matplotlib figure
fig = Figure(figsize=(8, 6), dpi=100)
ax = fig.add_subplot(111)

# Create a canvas for Matplotlib figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Bind drop event
root.drop_target_register(DND_FILES)

# Function to handle file drop
def handle_drop(event):
    file_path = event.data
    if file_path:
        detect_objects(file_path)

root.dnd_bind('<<Drop>>', handle_drop)

# Create Upload button
upload_button = tk.Button(root, text="Upload Image".upper(), command=handle_upload, bg="deep sky blue", font=("Helvetica", 12, "bold"))
upload_button.pack(side=tk.BOTTOM, pady=10)

# Run Tkinter main loop
root.mainloop()

