from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import sys

# --- Usage ---
# python augment.py Amine
# python augment.py Outmane
# -------------------------

# Get folder name from command line argument
if len(sys.argv) < 2:
    print("Usage: python augment.py <folder_name>")
    sys.exit(1)

folder_name = sys.argv[1]
folder = os.path.join("dataset", folder_name)

if not os.path.exists(folder):
    print(f"Error: folder '{folder}' does not exist in dataset/")
    sys.exit(1)

augment = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

TARGET_COUNT = 30  # Nombre d'images à produire par classe

images = os.listdir(folder)
images = [img for img in images if img.lower().endswith(("jpg", "png", "jpeg"))]

save_dir = folder  # On réécrit dans le même dossier

for img_name in images:
    img_path = os.path.join(folder, img_name)
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in augment.flow(x, batch_size=1, save_to_dir=save_dir,
                              save_prefix="aug", save_format="jpg"):
        i += 1
        if i >= TARGET_COUNT:
            break

print(f"Dataset augmenté dans : {folder}")
