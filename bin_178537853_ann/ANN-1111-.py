from ultralytics import YOLO
import cv2
import os
# Load YOLO model
model = YOLO("yolov8n.pt")
# Image path
image_path = r"C:\Users\Sujal\Downloads\car.jpeg"
# Check image
if not os.path.exists(image_path):
 print(" Image not found.")
 exit()
image = cv2.imread(image_path)
# Run detection
results = model(image)
print("\nObject Detection Successful!\n")
# Image information
h, w, c = image.shape
print(f"Image Resolution: {w} × {h} pixels\n") 

# Object counting
object_counts = {}
for result in results:
 for box in result.boxes:
    cls_id = int(box.cls[0])
 label = model.names[cls_id]
 object_counts[label] = object_counts.get(label, 0) + 1
print("Objects Detected:")
for obj, count in object_counts.items():
 print(f"• {obj.capitalize():12}: {count}")