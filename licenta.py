
from google.colab import drive
drive.mount('/content/drive')

!cp -r /content/drive/MyDrive/Alzheimer_Project/OriginalDataset /content/OriginalDataset

!cp -r /content/drive/MyDrive/Alzheimer_Project/FinalDataset /content/SplitDataset

"""# Importing necessary libraries"""

import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight

"""# Visualize dataset"""

dataset_dir='/content/OriginalDataset'

class_count={}

for class_name in os.listdir(dataset_dir):
  class_count[class_name]=len(os.listdir(os.path.join(dataset_dir,class_name)))
  print(f"Clasa {class_name}: {class_count[class_name]} imagini")


plt.figure(figsize=(10, 5))
sns.barplot(x=list(class_count.keys()), y=list(class_count.values()))
plt.xlabel("Clase")
plt.ylabel("Număr imagini")
plt.title("Numări imagini pe clase")
plt.show()

"""# Spliting the dataset"""

ORIGINAL_DATASET_DIR = "/content/OriginalDataset"
SPLIT_OUTPUT_DIR = "/content/SplitDataset"
random.seed(42)

if os.path.exists(SPLIT_OUTPUT_DIR):
    shutil.rmtree(SPLIT_OUTPUT_DIR)

def split_dataset(source_dir, dest_dir):
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images)

        total = len(images)
        test_count = int(0.2 * total)
        remaining = images[test_count:]
        val_count = int(0.2 * len(remaining))

        split_dict = {
            'test': images[:test_count],
            'val': remaining[:val_count],
            'train': remaining[val_count:]
        }

        for split, imgs in split_dict.items():
            split_cls_path = os.path.join(dest_dir, split, cls)
            os.makedirs(split_cls_path, exist_ok=True)
            for img in imgs:
                src = os.path.join(cls_path, img)
                dst = os.path.join(split_cls_path, img)
                shutil.copy(src, dst)

split_dataset(ORIGINAL_DATASET_DIR, SPLIT_OUTPUT_DIR)
print("Setul de date a fost împățit în train/val/test.")

"""# Visualize split dataset"""

SPLIT_OUTPUT_DIR='/content/SplitDataset'
def count_images_per_class(dataset_dir):
  counts={}
  for class_name in os.listdir(dataset_dir):
    counts[class_name]=len(os.listdir(os.path.join(dataset_dir,class_name)))
  return counts

train_counts=count_images_per_class(os.path.join(SPLIT_OUTPUT_DIR,'train'))
val_counts=count_images_per_class(os.path.join(SPLIT_OUTPUT_DIR,'val'))
test_counts=count_images_per_class(os.path.join(SPLIT_OUTPUT_DIR,'test'))

print("Număr antrenare:")
print(train_counts)
print("\nNumăr validare:")
print(val_counts)
print("\nNumăr test:")
print(test_counts)

df=pd.DataFrame({'Train':train_counts,'Val':val_counts,'Test':test_counts})
df.plot(kind='bar', figsize=(10, 5))
plt.xlabel("Clase")
plt.ylabel("Număr de imagini")
plt.title("Număr de imagini pe clasă")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=30)
plt.show()

"""# Augumentig dataset - just 2 classes"""

input_dir = "/content/SplitDataset/train"
output_dir = "/content/SplitDataset/train_augmented"


classes = ["ModerateDemented", "MildDemented"]
num_new_images = 1000


if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)


augmentor = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    fill_mode='nearest'
)

original_counts = {}
final_counts = {}

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    images = os.listdir(class_path)
    current_count = len(images)
    original_counts[class_name] = current_count

    for img_file in images:
        shutil.copy(os.path.join(class_path, img_file), output_class_path)

    if class_name in classes:
        num_images = len(images)
        copies_per_image = num_new_images // num_images
        extra = num_new_images % num_images


        aug_counter = 0


        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = load_img(img_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            for i in range(copies_per_image):
                aug_img = next(augmentor.flow(x, batch_size=1))[0].astype(np.uint8)
                save_path = os.path.join(output_class_path, f"aug_{aug_counter}_{img_name}")
                tf.keras.preprocessing.image.save_img(save_path, aug_img)
                aug_counter += 1

        idx_extra = 0
        while aug_counter < num_new_images:
            img_name = images[idx_extra % num_images]
            img_path = os.path.join(class_path, img_name)
            img = load_img(img_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            aug_img = next(augmentor.flow(x, batch_size=1))[0].astype(np.uint8)
            save_path = os.path.join(output_class_path, f"aug_extra_{aug_counter}_{img_name}")
            tf.keras.preprocessing.image.save_img(save_path, aug_img)
            aug_counter += 1
            idx_extra += 1

    final_counts[class_name] = len(os.listdir(output_class_path))

print("\nDISTRIBUȚIE FINALĂ:")
for class_name in final_counts:
    print(f"{class_name}: {final_counts[class_name]} imagini")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=list(original_counts.keys()), y=list(original_counts.values()))
plt.title("Înainte de augmentare")
plt.ylabel("Număr imagini")
plt.xticks(rotation=30)

plt.subplot(1, 2, 2)
sns.barplot(x=list(final_counts.keys()), y=list(final_counts.values()))
plt.title("După augmentare")
plt.ylabel("Număr imagini")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

SPLIT_OUTPUT_DIR='/content/SplitDataset'
def count_images_per_class(dataset_dir):
  counts={}
  for class_name in os.listdir(dataset_dir):
    counts[class_name]=len(os.listdir(os.path.join(dataset_dir,class_name)))
  return counts

train_counts=count_images_per_class(os.path.join(SPLIT_OUTPUT_DIR,'train_augmented'))
val_counts=count_images_per_class(os.path.join(SPLIT_OUTPUT_DIR,'val'))
test_counts=count_images_per_class(os.path.join(SPLIT_OUTPUT_DIR,'test'))

print("Număr antrenare:")
print(train_counts)
print("\nNumăr validare:")
print(val_counts)
print("\nNumăr test:")
print(test_counts)

df=pd.DataFrame({'Train':train_counts,'Val':val_counts,'Test':test_counts})
df.plot(kind='bar', figsize=(10, 5))
plt.title("Număr de imagini pe clasă")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=30)
plt.show()

"""# Show augmented images"""

def show_original_and_augmented(class_name, num_aug):

    output_dir = "/content/SplitDataset/train_augmented"

    output_class_path = os.path.join(output_dir, class_name)
    all_images = sorted(os.listdir(output_class_path))


    original_image_name = [img for img in all_images if not img.startswith('aug_')][35]
    print(original_image_name)

    augmented_images = [img for img in all_images if img.startswith('aug_') and original_image_name in img]

    augmented_images = sorted(augmented_images)[:num_aug]

    imgs = [load_img(os.path.join(output_class_path, original_image_name))]
    for aug_img_name in augmented_images:
        aug_img = load_img(os.path.join(output_class_path, aug_img_name))
        print(aug_img_name)
        imgs.append(aug_img)

    plt.figure(figsize=(4 * (num_aug + 1), 4))
    for idx, img in enumerate(imgs):
        plt.subplot(1, num_aug + 1, idx + 1)
        plt.imshow(img)
        if idx == 0:
            plt.title("Original")
        else:
            plt.title(f"Augmentat {idx}")
        plt.axis('off')
    plt.suptitle(f"Clasa: {class_name} — Exemplu augmentare", fontsize=16, y=1.05)
    plt.show()



show_original_and_augmented("MildDemented", 2)
show_original_and_augmented("ModerateDemented", 2)

image_folder = "/content/SplitDataset/train/MildDemented"

num_augmented = 5

all_images = [f for f in os.listdir(image_folder) if not f.startswith('aug_')]
random_image_name = random.choice(all_images)
image_path = os.path.join(image_folder, random_image_name)



augumentor = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255
    )

img = load_img(image_path)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

augmented_images = []
gen = augmentor.flow(x, batch_size=1)

for i in range(num_augmented):
    augmented = next(gen)[0].astype(np.uint8)
    augmented_images.append(augmented)

plt.figure(figsize=(3*(num_augmented+1), 4))

plt.subplot(1, num_augmented + 1, 1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

for i, aug_img in enumerate(augmented_images):
    plt.subplot(1, num_augmented + 1, i + 2)
    plt.imshow(aug_img)
    plt.title(f"Augmentat {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

"""# Adding class weights"""

y_train_labels = []

steps = len(train_data)
for step in tqdm(range(steps)):
    images, labels = next(train_data)
    y = np.argmax(labels, axis=1)
    y_train_labels.extend(y)

y_train_labels = np.array(y_train_labels)


d = {}
for label in y_train_labels:
    d[label] = d.get(label, 0) + 1

print("\nDistribuția pe clase:")
for label, number in sorted(d.items()):
    print(f"Clasa {label}: {number} imagini")


weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

class_weights = dict(zip(np.unique(y_train_labels), weights))

print("\nPonderi de clasă calculate:")
for label, weight in sorted(class_weights.items()):
    print(f"Clasa {label} {class_labels[label]}: {weight:.4f}")

TRAIN_PATH="/content/SplitDataset/train_augmented"
TRAIN="/content/SplitDataset/train"
VAL_PATH="/content/SplitDataset/val"
TEST_PATH="/content/SplitDataset/test"

class_labels = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

"""# Evaluate Model"""

def plot_confusion_matrix(true_class_labels, predicted_class_labels):
    cm = confusion_matrix(true_class_labels, predicted_class_labels)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicții")
    plt.ylabel("Etichete reale")
    plt.title("Matrice de confuzie")
    plt.show()

def evaluate_model(model, test_data, history=None):

    loss, acc = model.evaluate(test_data)
    print(f"\nAcuratețea pe test: \033[1m{acc:.4f}\033[0m")
    print(f"Pierderea pe test: \033[1m{loss:.4f}\033[0m")

    if history:

      plt.figure(figsize=(12, 5))

      plt.subplot(1, 2, 1)
      plt.plot(history.history['accuracy'], label='Acuratețe pe antrenament')
      plt.plot(history.history['val_accuracy'], label='Acuratețe pe validare')
      plt.title("Acuratețe pe antrenament și validare")
      plt.xlabel("Epoca")
      plt.ylabel("Acuratețe")
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(history.history['loss'], label='Pierdere pe antrenament')
      plt.plot(history.history['val_loss'], label='Pierdere pe validare')
      plt.title("Pierdere pe antrenament și validare")
      plt.xlabel("Epoca")
      plt.ylabel("Pierdere")
      plt.legend()

      plt.tight_layout()
      plt.show()



    true_class_labels = test_data.classes

    probabilities = model.predict(test_data)

    predicted_class_labels = np.argmax(probabilities, axis=1)


    print("\nRaportul de clasificare:\n")
    print(classification_report(true_class_labels, predicted_class_labels, target_names=class_labels))

    print(true_class_labels)
    print(predicted_class_labels)
    plot_confusion_matrix(true_class_labels, predicted_class_labels)

"""# Real class vs Predicted class"""

def choose_random_images(test_data, num_images):
  images=[]
  labels=[]

  while len(images) < num_images:
     batch_idx= np.random.randint(0, len(test_data))
     batch_images, batch_labels = test_data[batch_idx]

     idx=np.random.randint(0, len(batch_images))
     image=batch_images[idx]
     images.append(image)
     label=np.argmax(batch_labels[idx])
     labels.append(label)

  return images, labels

def plot_sample_predictions(images, true_class_labels_names, predicted_class_labels_names, prediction_probs):

    plt.figure(figsize=(10, 5))

    for i in range(len(images)):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        is_correct= true_class_labels_names[i] == predicted_class_labels_names[i]
        color = 'green' if is_correct else 'red'

        prediction=np.max(prediction_probs[i])
        plt.title(f"Real: {true_class_labels_names[i]}\nPrezis: {predicted_class_labels_names[i]}\n Probabilitate: {prediction*100:.2f}%", fontsize=8, color=color)

    plt.suptitle("Exemple: Clasă reală vs prezisă", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_correct_predictions_bar_chart(true_class_labels, predicted_class_labels):

    correct_per_class = Counter()
    total_per_class = Counter()

    for true_class, pred_class in zip(true_class_labels, predicted_class_labels):
        if true_class == pred_class:
            correct_per_class[true_class] += 1

        total_per_class[true_class] += 1

    labels_short = [class_labels[i] for i in range(len(class_labels))]
    correct_counts = [correct_per_class[i] for i in range(len(class_labels))]
    total_counts = [total_per_class[i] for i in range(len(class_labels))]


    plt.figure(figsize=(10,6))
    bar_width = 0.4
    indices = np.arange(len(class_labels))

    plt.bar(indices-bar_width/2, total_counts, width=bar_width, label='Total', color='steelblue')
    plt.bar(indices+bar_width/2, correct_counts, width=bar_width*0.7, label='Corecte', color='green')

    plt.xticks(indices, labels_short)
    plt.ylabel("Număr imagini")
    plt.title("Clasificări corecte per clasă")
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, test_data):

    images, true_class_labels= choose_random_images(test_data, 6)

    prediction_probs=model.predict(np.array(images))

    print(prediction_probs)

    predicted_class_idx=np.argmax(prediction_probs, axis=1)

    true_class_labels_names=[class_labels[i] for i in true_class_labels]
    predicted_class_labels_names=[class_labels[i] for i in predicted_class_idx]


    plot_sample_predictions(images, true_class_labels_names, predicted_class_labels_names, prediction_probs)

    all_true_class_labels=test_data.classes

    all_predicted_class_labels=np.argmax(model.predict(test_data), axis=1)

    plot_correct_predictions_bar_chart(all_true_class_labels, all_predicted_class_labels)

def load_data_generators(img_size, batch_size, color_mode):

    train_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255
    )



    test_generator = ImageDataGenerator(rescale=1./255)

    train_data = train_generator.flow_from_directory(
        directory=TRAIN_PATH,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode
    )

    val_data = test_generator.flow_from_directory(
        directory=VAL_PATH,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode

    )

    test_data = test_generator.flow_from_directory(
        directory=TEST_PATH,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False
    )

    return train_data, val_data, test_data

def load_data_generators1(img_size, batch_size, color_mode):

    train_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.1,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255
    )


    test_generator = ImageDataGenerator(rescale=1./255)

    train_data = train_generator.flow_from_directory(
        directory=TRAIN,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode
    )

    val_data = test_generator.flow_from_directory(
        directory=VAL_PATH,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode

    )

    test_data = test_generator.flow_from_directory(
        directory=TEST_PATH,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False
    )

    return train_data, val_data, test_data

"""# CNN Arhitecture
*   IMG_SIZE=128
*   BATCH_SIZE=32
*   EPOCHS=20
*   NUM_CHANNELS=1
"""

IMG_SIZE=128
BATCH_SIZE=32
EPOCHS=20
NUM_CHANNELS=1

def build_CNN_model(input_shape):

    CNN_model = models.Sequential()

    CNN_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    CNN_model.add(layers.MaxPooling2D(2, 2))

    CNN_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    CNN_model.add(layers.MaxPooling2D(2, 2))

    CNN_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    CNN_model.add(layers.MaxPooling2D(2, 2))

    CNN_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    CNN_model.add(layers.MaxPooling2D(2, 2))

    CNN_model.add(layers.Flatten())

    CNN_model.add(layers.Dense(256, activation='relu'))
    CNN_model.add(layers.Dropout(0.4))

    CNN_model.add(layers.Dense(4, activation='softmax'))


    CNN_model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return CNN_model

def build_CNN_model(input_shape):
    CNN_model = models.Sequential()


    CNN_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    CNN_model.add(layers.MaxPooling2D(2, 2))

    CNN_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    CNN_model.add(layers.MaxPooling2D(2, 2))

    CNN_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    CNN_model.add(layers.MaxPooling2D(2, 2))

    CNN_model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    CNN_model.add(layers.MaxPooling2D(2, 2))

    CNN_model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    CNN_model.add(layers.MaxPooling2D(2, 2))


    CNN_model.add(layers.Flatten())

    CNN_model.add(layers.Dense(256, activation='relu'))
    CNN_model.add(layers.Dropout(0.4))

    CNN_model.add(layers.Dense(4, activation='softmax'))


    CNN_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return CNN_model

model=build_CNN_model2((IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
model.summary()

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, 'grayscale')
NUM_CHANNELS

check_point= ModelCheckpoint(filepath='/content/CNN_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

CNN_model=build_CNN_model(input_shape)

CNN_history=CNN_model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[check_point])

evaluate_model(CNN_model, test_data, CNN_history)

visualize_predictions(CNN_model, test_data)

model=build_CNN_model2((IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
model.summary()

check_point= ModelCheckpoint(filepath='/content/CNN_model_nou.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

CNN_model2=build_CNN_model2(input_shape)

CNN_history2=CNN_model2.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[check_point])

evaluate_model(CNN_model2, test_data, CNN_history2)
visualize_predictions(CNN_model2, test_data)

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

CNN_model2=build_CNN_model2(input_shape)

CNN_history2=CNN_model2.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[check_point], class_weight=class_weights)

evaluate_model(CNN_model2, test_data, CNN_history2)
visualize_predictions(CNN_model2, test_data)

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, 'grayscale')

check_point= ModelCheckpoint(filepath='/content/CNN_model2.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

CNN_model2=build_CNN_model(input_shape)

CNN_history2=CNN_model2.fit(train_data, validation_data=val_data, epochs=30, callbacks=[check_point], class_weight=class_weights)

evaluate_model(CNN_model2, test_data, CNN_history2)

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

CNN_model3=build_CNN_model(input_shape)

CNN_history3=CNN_model3.fit(train_data, validation_data=val_data, epochs=30, callbacks=[check_point])

evaluate_model(CNN_model3, test_data, CNN_history3)

visualize_predictions(CNN_model3, test_data)

"""# VGG16
*   IMG_SIZE=224
*   BATCH_SIZE=32
*   EPOCHS=30
*   NUM_CHANNELS=3
*   LEARNING_RATE=0.0001
"""

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
NUM_CHANNELS=3
LEARNING_RATE=0.0001

"""# With Imagenet"""

def build_VGG16_model(input_shape):

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    base_model.trainable = False

    VGG16_model = models.Sequential()
    VGG16_model.add(base_model)
    VGG16_model.add(layers.Flatten())
    VGG16_model.add(layers.Dense(256, activation='relu'))
    VGG16_model.add(layers.Dropout(0.4))
    VGG16_model.add(layers.Dense(4, activation='softmax'))



    VGG16_model.compile(optimizer=Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    VGG16_model.summary()
    return VGG16_model

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, 'rgb')
IMG_SIZE

check_point= ModelCheckpoint(filepath='/content/VGG16_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

VGG16_model = build_VGG16_model(input_shape)

VGG16_history = VGG16_model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[check_point])

evaluate_model(VGG16_model, test_data, VGG16_history)

visualize_predictions(VGG16_model, test_data)

VGG16_model.summary()

"""# Without Imagenet"""

def build_VGG16_model2(input_shape):

    base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)

    VGG16_model2 = models.Sequential()
    VGG16_model2.add(base_model)
    VGG16_model2.add(layers.GlobalAveragePooling2D())
    VGG16_model2.add(layers.Dense(256, activation='relu'))
    VGG16_model2.add(layers.Dropout(0.4))
    VGG16_model2.add(layers.Dense(4, activation='softmax'))



    VGG16_model2.compile(optimizer=Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    return VGG16_model2

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, 'rgb')
IMG_SIZE

model_checkpoint = ModelCheckpoint(filepath='/content/VGG16_model2.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

VGG16_model2 = build_VGG16_model2(input_shape)

VGG16_history2 = VGG16_model2.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[model_checkpoint])

evaluate_model(VGG16_model2, test_data, VGG16_history2)

visualize_predictions(VGG16_model2, test_data)

"""#   INCEPTIONV3
*   IMG_SIZE=299
*   BATCH_SIZE=32
*   EPOCHS=30
*   NUM_CHANNELS=3
*   LEARNING_RATE=0.0001


"""

IMG_SIZE = 299
BATCH_SIZE = 32
EPOCHS = 30
NUM_CHANNELS=3
LEARNING_RATE=0.0001

"""# Without Imagenet"""

def build_InceptionV3_model1(input_shape):

   base_model = InceptionV3(weights=None, include_top=False, input_shape=input_shape)

   InceptionV3_model = models.Sequential()
   InceptionV3_model.add(base_model)
   InceptionV3_model.add(layers.GlobalAveragePooling2D())
   InceptionV3_model.add(layers.Dense(256, activation='relu'))
   InceptionV3_model.add(layers.Dropout(0.4))
   InceptionV3_model.add(layers.Dense(4, activation='softmax'))



   InceptionV3_model.compile(optimizer=Adam(learning_rate=0.0001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

   return InceptionV3_model

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

check_point= ModelCheckpoint(filepath='/content/InceptionV3_1_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
InceptionV3_model1 = build_InceptionV3_model1(input_shape)
InceptionV3_history = InceptionV3_model1.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[check_point])

evaluate_model(InceptionV3_model1, test_data, InceptionV3_history)

visualize_predictions(InceptionV3_model1, test_data)

InceptionV3_model1.summary()

"""# With Imagenet"""

def build_InceptionV3_model_imagenet(input_shape):

   base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
   base_model.trainable = False

   InceptionV3_model = models.Sequential()
   InceptionV3_model.add(base_model)
   InceptionV3_model.add(layers.GlobalAveragePooling2D())
   InceptionV3_model.add(layers.Dense(256, activation='relu'))
   InceptionV3_model.add(layers.Dropout(0.4))
   InceptionV3_model.add(layers.Dense(4, activation='softmax'))



   InceptionV3_model.compile(optimizer=Adam(learning_rate=0.0001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

   return InceptionV3_model

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

model_checkpoint = ModelCheckpoint(filepath='/content/InceptionV3_model_imagenet.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
InceptionV3_model_imagenet = build_InceptionV3_model_imagenet(input_shape)
InceptionV3_history_imagenet = InceptionV3_model_imagenet.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[model_checkpoint])

evaluate_model(InceptionV3_model_imagenet, test_data, InceptionV3_history_imagenet)

visualize_predictions(InceptionV3_model_imagenet, test_data)

def build_InceptionV3_model_imagenet2(input_shape):

   base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
   base_model.trainable = False

   InceptionV3_model = models.Sequential()
   InceptionV3_model.add(base_model)
   InceptionV3_model.add(layers.GlobalAveragePooling2D())
   InceptionV3_model.add(layers.Dense(256))
   InceptionV3_model.add(layers.BatchNormalization())
   InceptionV3_model.add(layers.Activation('relu'))
   InceptionV3_model.add(layers.Dropout(0.4))
   InceptionV3_model.add(layers.Dense(4, activation='softmax'))



   InceptionV3_model.compile(optimizer=Adam(learning_rate=0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

   return InceptionV3_model

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

model_checkpoint = ModelCheckpoint(filepath='/content/InceptionV3_2_model_imagenet.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
lr_scheduler= ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
InceptionV3_model2_imagenet = build_InceptionV3_model_imagenet2(input_shape)
InceptionV3_history2_imagenet = InceptionV3_model2_imagenet.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[model_checkpoint, lr_scheduler])

evaluate_model(InceptionV3_model2_imagenet, test_data, InceptionV3_history2_imagenet)

visualize_predictions(InceptionV3_model2_imagenet, test_data)

"""# Am adaugat batchnormalization si learning rate scheduler si am crescut rata la 0.001"""

def build_InceptionV3_model2(input_shape):

   base_model = InceptionV3(weights=None, include_top=False, input_shape=input_shape)

   InceptionV3_model2 = models.Sequential()
   InceptionV3_model2.add(base_model)
   InceptionV3_model2.add(layers.GlobalAveragePooling2D())
   InceptionV3_model2.add(layers.Dense(256))
   InceptionV3_model2.add(layers.BatchNormalization())
   InceptionV3_model2.add(layers.Activation('relu'))
   InceptionV3_model2.add(layers.Dropout(0.4))
   InceptionV3_model2.add(layers.Dense(4, activation='softmax'))



   InceptionV3_model2.compile(optimizer=Adam(learning_rate=0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

   return InceptionV3_model2

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

model_checkpoint = ModelCheckpoint(filepath='/content/InceptionV3_2_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
lr_scheduler= ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
InceptionV3_model2 = build_InceptionV3_model2(input_shape)
InceptionV3_history2 = InceptionV3_model2.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[model_checkpoint, lr_scheduler])

evaluate_model(InceptionV3_model2, test_data, InceptionV3_history2)

visualize_predictions(InceptionV3_model2, test_data)

def build_InceptionV3_model_nou(input_shape):

   base_model = InceptionV3(weights=None, include_top=False, input_shape=input_shape)

   InceptionV3_model2 = models.Sequential()
   InceptionV3_model2.add(base_model)
   InceptionV3_model2.add(layers.GlobalAveragePooling2D())
   InceptionV3_model2.add(layers.Dense(256))
   #InceptionV3_model2.add(layers.BatchNormalization())
   InceptionV3_model2.add(layers.Activation('relu'))
   InceptionV3_model2.add(layers.Dropout(0.4))
   InceptionV3_model2.add(layers.Dense(4, activation='softmax'))



   InceptionV3_model2.compile(optimizer=Adam(learning_rate=0.0001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

   return InceptionV3_model2

train_data, val_data, test_data= load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")

check_point= ModelCheckpoint(filepath='/content/InceptionV3_model_nou.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
lr_scheduler= ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
InceptionV3_model_nou = build_InceptionV3_model_nou(input_shape)
InceptionV3_history_nou = InceptionV3_model_nou.fit(train_data, validation_data=val_data, epochs=50, callbacks=[check_point])

evaluate_model(InceptionV3_model_nou, test_data, InceptionV3_history_nou)

"""# Am adauga încă un strat cu 512 neuroni, unul de batch norma si unul de dropout"""

def build_InceptionV3_model3(input_shape):

   base_model = InceptionV3(weights=None, include_top=False, input_shape=input_shape)

   InceptionV3_model = models.Sequential()
   InceptionV3_model.add(base_model)
   InceptionV3_model.add(layers.GlobalAveragePooling2D())
   InceptionV3_model.add(layers.Dense(512))
   InceptionV3_model.add(layers.BatchNormalization())
   InceptionV3_model.add(layers.Activation('relu'))
   InceptionV3_model.add(layers.Dropout(0.5))
   InceptionV3_model.add(layers.Dense(256))
   InceptionV3_model.add(layers.BatchNormalization())
   InceptionV3_model.add(layers.Activation('relu'))
   InceptionV3_model.add(layers.Dropout(0.4))
   InceptionV3_model.add(layers.Dense(4, activation='softmax'))



   InceptionV3_model.compile(optimizer=Adam(learning_rate=0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
   InceptionV3_model.summary()

   return InceptionV3_model

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

model_checkpoint = ModelCheckpoint(filepath='/content/InceptionV3_3_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
lr_scheduler= ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
InceptionV3_model3 = build_InceptionV3_model3(input_shape)
InceptionV3_history3 = InceptionV3_model3.fit(train_data, validation_data=val_data, epochs=50, callbacks=[model_checkpoint, lr_scheduler])

evaluate_model(InceptionV3_model3, test_data, InceptionV3_history3)

visualize_predictions(InceptionV3_model3, test_data)

"""# Acelasi model optimizat dar rata la 0.0001"""

def build_InceptionV3_model_9(input_shape):

   base_model = InceptionV3(weights=None, include_top=False, input_shape=input_shape)

   InceptionV3_model = models.Sequential()
   InceptionV3_model.add(base_model)
   InceptionV3_model.add(layers.GlobalAveragePooling2D())
   InceptionV3_model.add(layers.Dense(512))
   InceptionV3_model.add(layers.BatchNormalization())
   InceptionV3_model.add(layers.Activation('relu'))
   InceptionV3_model.add(layers.Dropout(0.5))
   InceptionV3_model.add(layers.Dense(256))
   InceptionV3_model.add(layers.BatchNormalization())
   InceptionV3_model.add(layers.Activation('relu'))
   InceptionV3_model.add(layers.Dropout(0.4))
   InceptionV3_model.add(layers.Dense(4, activation='softmax'))



   InceptionV3_model.compile(optimizer=Adam(learning_rate=0.0001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
   InceptionV3_model.summary()

   return InceptionV3_model

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

lr_scheduler= ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )
model_checkpoint = ModelCheckpoint(filepath='/content/InceptionV3_model_9.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
InceptionV3_model_9 = build_InceptionV3_model_9(input_shape)
InceptionV3_history_9 = InceptionV3_model_9.fit(train_data, validation_data=val_data, epochs=50, callbacks=[model_checkpoint, lr_scheduler])

evaluate_model(InceptionV3_model_9, test_data, InceptionV3_history_9)
visualize_predictions(InceptionV3_model_9, test_data)

"""# InceptionV3 + class weights"""

def build_InceptionV3_model4(input_shape):

   base_model = InceptionV3(weights=None, include_top=False, input_shape=input_shape)

   InceptionV4_model = models.Sequential()
   InceptionV4_model.add(base_model)
   InceptionV4_model.add(layers.GlobalAveragePooling2D())
   InceptionV4_model.add(layers.Dense(512))
   InceptionV4_model.add(layers.BatchNormalization())
   InceptionV4_model.add(layers.Activation('relu'))
   InceptionV4_model.add(layers.Dropout(0.5))
   InceptionV4_model.add(layers.Dense(256))
   InceptionV4_model.add(layers.BatchNormalization())
   InceptionV4_model.add(layers.Activation('relu'))
   InceptionV4_model.add(layers.Dropout(0.4))
   InceptionV4_model.add(layers.Dense(4, activation='softmax'))



   InceptionV4_model.compile(optimizer=Adam(learning_rate=0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
   InceptionV4_model.summary()

   return InceptionV4_model

model_checkpoint = ModelCheckpoint(filepath='/content/InceptionV3_model4_imagenet.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
lr_scheduler= ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
InceptionV4_model = build_InceptionV3_model4(input_shape)
InceptionV4_history = InceptionV4_model.fit(train_data, validation_data=val_data, epochs=50, callbacks=[model_checkpoint, lr_scheduler], class_weight=class_weights)

evaluate_model(InceptionV4_model, test_data, InceptionV4_history)
visualize_predictions(InceptionV4_model, test_data)

"""# Inception + dataset_original fără class_weights




"""

def build_InceptionV3_model5(input_shape):

   base_model = InceptionV3(weights=None, include_top=False, input_shape=input_shape)
   InceptionV4_model = models.Sequential()
   InceptionV4_model.add(base_model)
   InceptionV4_model.add(layers.GlobalAveragePooling2D())
   InceptionV4_model.add(layers.Dense(512))
   InceptionV4_model.add(layers.BatchNormalization())
   InceptionV4_model.add(layers.Activation('relu'))
   InceptionV4_model.add(layers.Dropout(0.5))
   InceptionV4_model.add(layers.Dense(256, activation='relu'))
   InceptionV4_model.add(layers.BatchNormalization())
   InceptionV4_model.add(layers.Activation('relu'))
   InceptionV4_model.add(layers.Dropout(0.4))
   InceptionV4_model.add(layers.Dense(4, activation='softmax'))

   InceptionV4_model.compile(optimizer=Adam(learning_rate=0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
   InceptionV4_model.summary()

   return InceptionV4_model

train_data, val_data, test_data = load_data_generators1(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

lr_scheduler= ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
InceptionV3_model = build_InceptionV3_model5(input_shape)
InceptionV3_history = InceptionV3_model.fit(train_data, validation_data=val_data, epochs=30, callbacks=[lr_scheduler])

evaluate_model(InceptionV3_model, test_data, InceptionV3_history)
visualize_predictions(InceptionV3_model, test_data)

"""# Inception+ dataset original+ class_weights"""

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
InceptionV3_model = build_InceptionV3_model5(input_shape)
InceptionV3_history = InceptionV3_model.fit(train_data, validation_data=val_data, epochs=30, callbacks=[lr_scheduler], class_weight=class_weights)

evaluate_model(InceptionV3_model, test_data, InceptionV3_history)
visualize_predictions(InceptionV3_model, test_data)

"""# RESNET50 Arhitecture
*   IMG_SIZE=224
*   BATCH_SIZE=32
*   EPOCHS=30
*   NUM_CHANNELS=3
*   LEARNING_RATE=0.0001

"""

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
NUM_CHANNELS=3
LEARNING_RATE=0.0001

"""# ResNet50 Transfer Learning"""

def build_ResNet50_imagenet(input_shape):

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    ResNet50_model = models.Sequential()
    ResNet50_model.add(base_model)
    ResNet50_model.add(layers.GlobalAveragePooling2D())
    ResNet50_model.add(layers.Dense(256, activation='relu'))
    ResNet50_model.add(layers.Dropout(0.4))
    ResNet50_model.add(layers.Dense(4, activation='softmax'))

    ResNet50_model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return ResNet50_model

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

check_point= ModelCheckpoint(filepath='/content/ResNet50_model_imagenet.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
ResNet50_model_imagenet = build_ResNet50_imagenet(input_shape)

ResNet50_history_imagenet = ResNet50_model_imagenet.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[check_point])

evaluate_model(ResNet50_model_imagenet, test_data, ResNet50_history_imagenet)

visualize_predictions(ResNet50_model_imagenet, test_data)

"""# Toată Arhitectura"""

def build_ResNet50(input_shape):

    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    ResNet50_model = models.Sequential()
    ResNet50_model.add(base_model)
    ResNet50_model.add(layers.GlobalAveragePooling2D())
    ResNet50_model.add(layers.Dense(256, activation='relu'))
    ResNet50_model.add(layers.Dropout(0.4))
    ResNet50_model.add(layers.Dense(4, activation='softmax'))

    ResNet50_model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return ResNet50_model

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

check_point= ModelCheckpoint(filepath='/content/ResNet50_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
ResNet50_model = build_ResNet50(input_shape)

ResNet50_history = ResNet50_model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[check_point])

evaluate_model(ResNet50_model, test_data, ResNet50_history)

visualize_predictions(ResNet50_model, test_data)

def build_ResNet50_2(input_shape):

    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    ResNet50_model2 = models.Sequential()
    ResNet50_model2.add(base_model)
    ResNet50_model2.add(layers.GlobalAveragePooling2D())
    ResNet50_model2.add(layers.Dense(256, activation='relu'))
    ResNet50_model2.add(layers.BatchNormalization())
    ResNet50_model2.add(layers.Dropout(0.4))
    ResNet50_model2.add(layers.Dense(4, activation='softmax'))

    ResNet50_model2.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return ResNet50_model2

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

check_point= ModelCheckpoint(filepath='/content/ResNet50_model2.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
ResNet50_model2 = build_ResNet50_2(input_shape)

ResNet50_history2 = ResNet50_model2.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[check_point, lr_scheduler])

evaluate_model(ResNet50_model2, test_data, ResNet50_history2)

visualize_predictions(ResNet50_model2, test_data)

"""# Planificator+ rata crescuta+ strat dens 512"""

def build_ResNet503(input_shape):

    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)

    ResNet50_model3 = models.Sequential()
    ResNet50_model3.add(base_model)
    ResNet50_model3.add(layers.GlobalAveragePooling2D())
    ResNet50_model3.add(layers.Dense(512, activation='relu'))
    ResNet50_model3.add(layers.BatchNormalization())
    ResNet50_model3.add(layers.Dropout(0.5))
    ResNet50_model3.add(layers.Dense(256, activation='relu'))
    ResNet50_model3.add(layers.BatchNormalization())
    ResNet50_model3.add(layers.Dropout(0.4))
    ResNet50_model3.add(layers.Dense(4, activation='softmax'))

    ResNet50_model3.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return ResNet50_model3

train_data, val_data, test_data = load_data_generators(IMG_SIZE, BATCH_SIZE, "rgb")
IMG_SIZE

check_point= ModelCheckpoint(filepath='/content/ResNet50_model4.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )

input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
ResNet50_model3 = build_ResNet503(input_shape)
ResNet50_history3 = ResNet50_model3.fit(train_data, validation_data=val_data, epochs=50, callbacks=[check_point, lr_scheduler])

evaluate_model(ResNet50_model3, test_data, ResNet50_history3)

visualize_predictions(ResNet50_model3, test_data)