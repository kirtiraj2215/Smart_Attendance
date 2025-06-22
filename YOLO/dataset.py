import os
import random
import shutil
import cv2
import numpy as np

def create_yolo_dataset(
    class_paths,
    base_folder, 
    val_split=0.2, 
    use_annotations=False, 
    annotation_folder=None
):
    """
    Advanced YOLO dataset creation with optional annotation support.
    
    Args:
        class_paths (dict): Dictionary of class names and their image paths
        base_folder (str): Base directory to save the YOLO dataset
        val_split (float): Fraction of images for validation
        use_annotations (bool): Use existing bounding box annotations
        annotation_folder (str): Path to annotation folder if using existing annotations
    """
    # Define YOLO-compatible paths
    train_images_dir = os.path.join(base_folder, "train", "images")
    train_labels_dir = os.path.join(base_folder, "train", "labels")
    val_images_dir = os.path.join(base_folder, "val", "images")
    val_labels_dir = os.path.join(base_folder, "val", "labels")

    # Create directories
    for folder in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(folder, exist_ok=True)

    # Prepare class mapping
    class_names = list(class_paths.keys())

    def process_class(
        class_id, 
        src_folder, 
        train_images_dir, 
        train_labels_dir, 
        val_images_dir, 
        val_labels_dir
    ):
        # Get image files
        image_files = [
            f for f in os.listdir(src_folder) 
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
        ]
        random.shuffle(image_files)

        # Split into train and validation sets
        split_index = int(len(image_files) * (1 - val_split))
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        # Process files for both train and validation sets
        for split, file_list, images_dir, labels_dir in [
            ('train', train_files, train_images_dir, train_labels_dir),
            ('val', val_files, val_images_dir, val_labels_dir)
        ]:
            for file in file_list:
                # Copy image
                src_image_path = os.path.join(src_folder, file)
                dst_image_path = os.path.join(images_dir, file)
                shutil.copy(src_image_path, dst_image_path)

                # Handle label creation
                label_filename = os.path.splitext(file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_filename)

                if use_annotations and annotation_folder:
                    # Look for corresponding annotation
                    ann_file = os.path.join(annotation_folder, label_filename)
                    if os.path.exists(ann_file):
                        shutil.copy(ann_file, label_path)
                    else:
                        # Create default full-image annotation if no specific annotation
                        create_default_annotation(dst_image_path, label_path, class_id)
                else:
                    # Create default full-image annotation
                    create_default_annotation(dst_image_path, label_path, class_id)

    # Process each class
    for class_name, class_path in class_paths.items():
        process_class(
            class_names.index(class_name), 
            class_path, 
            train_images_dir, 
            train_labels_dir, 
            val_images_dir, 
            val_labels_dir
        )

    # Create data.yaml
    data_yaml_path = os.path.join(base_folder, "data.yaml")
    with open(data_yaml_path, "w") as yaml_file:
        yaml_file.write(f"train: {train_images_dir}\n")
        yaml_file.write(f"val: {val_images_dir}\n")
        yaml_file.write(f"nc: {len(class_names)}\n")
        yaml_file.write(f"names: {class_names}\n")

    print(f"YOLO dataset created at {base_folder}")
    print(f"Classes: {class_names}")

def create_default_annotation(image_path, label_path, class_id):
    """
    Create a default full-image bounding box annotation
    """
    try:
        # Read image to get dimensions
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        # Create default annotation (centered, full image)
        with open(label_path, 'w') as f:
            # YOLO format: <class> <x_center> <y_center> <width> <height>
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Example usage
base_folder = r"C:\Users\CSE IIT BHILAI\Documents\cc\dataset3"
class_paths = {
    'msd': r"C:\Users\CSE IIT BHILAI\Documents\cc\datasets\yolo_dataset\train\ms_dhoni",
    'kapil_dev': r"C:\Users\CSE IIT BHILAI\Documents\cc\datasets\yolo_dataset\train\kapil_dev"
}

create_yolo_dataset(
    class_paths, 
    base_folder, 
    val_split=0.2,  # 20% validation split
    use_annotations=False  # Set to True if you have pre-existing annotations
)