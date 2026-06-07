import os
import cv2
import numpy as np

LABEL_MAP = {'rock': 0, 'paper': 1, 'scissors': 2}
LABEL_NAMES = ['Rock', 'Paper', 'Scissors']
IMAGE_SIZE = (64, 64)
CNN_IMAGE_SIZE = (64, 64)
MOBILENET_IMAGE_SIZE = (96, 96)
EFFICIENTNET_IMAGE_SIZE = (224, 224)
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def resolve_dataset_category_dir(folder_path, category):
    category_path = os.path.join(folder_path, category)
    if os.path.isdir(category_path):
        return category_path

    try:
        subdirs = [
            d for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))
        ]
    except FileNotFoundError:
        return category_path

    for subdir in subdirs:
        nested_path = os.path.join(folder_path, subdir, category)
        if os.path.isdir(nested_path):
            return nested_path

    return category_path


def iter_labeled_image_paths(folder_path):
    for category, label_idx in LABEL_MAP.items():
        category_path = resolve_dataset_category_dir(folder_path, category)
        if not os.path.isdir(category_path):
            print(f"[WARN] Missing folder for {category}: {category_path}")
            continue

        print(f"[LOAD] Loading images for {category}...")
        for filename in sorted(os.listdir(category_path)):
            if filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
                yield os.path.join(category_path, filename), label_idx


def load_raw_images_from_folder(folder_path):
    images = []
    labels = []

    for image_path, label_idx in iter_labeled_image_paths(folder_path):
        image = cv2.imread(image_path)
        if image is None:
            continue
        images.append(image)
        labels.append(label_idx)

    return images, np.array(labels, dtype=np.int32)


def preprocess_ml_image(image, image_size=IMAGE_SIZE):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    return resized.astype(np.float32) / 255.0


def preprocess_flatten_image(image, image_size=IMAGE_SIZE):
    processed = preprocess_ml_image(image, image_size=image_size)
    return processed.flatten()


def preprocess_cnn_image(image, image_size=CNN_IMAGE_SIZE):
    resized = cv2.resize(image, image_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def preprocess_mobilenet_image(image, image_size=MOBILENET_IMAGE_SIZE):
    resized = cv2.resize(image, image_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def preprocess_efficientnet_image(image, image_size=EFFICIENTNET_IMAGE_SIZE):
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
    resized = cv2.resize(image, image_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return efficientnet_preprocess_input(rgb.astype(np.float32))


class HOGFeatureExtractor:
    def __init__(self, image_size=IMAGE_SIZE, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from skimage.feature import hog
        features = []
        for image in X:
            processed = preprocess_ml_image(image, image_size=self.image_size)
            features.append(
                hog(
                    processed,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    block_norm='L2-Hys'
                )
            )
        return features


def load_flatten_dataset(folder_path, image_size=IMAGE_SIZE):
    raw_images, labels = load_raw_images_from_folder(folder_path)
    features = [preprocess_flatten_image(image, image_size=image_size) for image in raw_images]
    return np.array(features, dtype=np.float32), labels


def load_cnn_dataset(folder_path, image_size=CNN_IMAGE_SIZE):
    raw_images, labels = load_raw_images_from_folder(folder_path)
    features = [preprocess_cnn_image(image, image_size=image_size) for image in raw_images]
    return np.array(features, dtype=np.float32), labels


def load_mobilenet_dataset(folder_path, image_size=MOBILENET_IMAGE_SIZE):
    raw_images, labels = load_raw_images_from_folder(folder_path)
    features = [preprocess_mobilenet_image(image, image_size=image_size) for image in raw_images]
    return np.array(features, dtype=np.float32), labels


def load_efficientnet_dataset(folder_path, image_size=EFFICIENTNET_IMAGE_SIZE):
    raw_images, labels = load_raw_images_from_folder(folder_path)
    features = [preprocess_efficientnet_image(image, image_size=image_size) for image in raw_images]
    return np.array(features, dtype=np.float32), labels


def print_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, labels=list(range(len(LABEL_NAMES))), target_names=LABEL_NAMES, zero_division=0))
