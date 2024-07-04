import argparse
import os
import random

from PIL import Image


def create_id_map(id_file_path, num_ids):
    """Create a map of original to new IDs and vice versa."""
    id_map = {}
    with open(id_file_path, "w") as id_file:
        for original_id in range(num_ids):
            new_id = format(original_id, "05d")
            id_map[original_id] = new_id
            id_map[new_id] = original_id
            id_file.write(f"{original_id} {new_id}\n")
    return id_map


def save_image(img, output_path):
    """Save the image at the specified path."""
    img.save(output_path, "JPEG")


def process_images(input_folder, output_folder, train_ratio=0.7):
    """Process images and save them with new IDs in the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if not os.path.exists(os.path.join(output_folder, subfolder, "train")):
            os.makedirs(os.path.join(output_folder, subfolder, "train"))
        if not os.path.exists(os.path.join(output_folder, subfolder, "val")):
            os.makedirs(os.path.join(output_folder, subfolder, "val"))
        if os.path.isdir(subfolder_path):
            for fol in os.listdir(subfolder_path):
                if "3d_scan" in fol:
                    continue
                images_folder_path = os.path.join(subfolder_path, fol, "images")
                print(images_folder_path)
                nid = fol.split("_")[1]
                if os.path.exists(images_folder_path):
                    train_folder = os.path.join(output_folder, subfolder, "train")
                    val_folder = os.path.join(output_folder, subfolder, "val")

                    if not os.path.exists(train_folder):
                        os.makedirs(train_folder)
                    if not os.path.exists(val_folder):
                        os.makedirs(val_folder)

                    image_files = [
                        f
                        for f in os.listdir(images_folder_path)
                        if f.endswith(".jpg") and ("downsampled" not in f)
                    ]
                    random.shuffle(image_files)
                    split_index = int(len(image_files) * train_ratio)
                    train_files = image_files[:split_index]
                    val_files = image_files[split_index:]

                    for image_file in train_files:
                        original_id = int(os.path.splitext(image_file)[0])
                        # if original_id in id_map:
                        new_id = nid
                        img_path = os.path.join(images_folder_path, image_file)
                        img = Image.open(img_path)

                        new_img_path = os.path.join(
                            train_folder, f"{new_id}_{original_id:03d}.jpg"
                        )
                        save_image(img, new_img_path)

                    for image_file in val_files:
                        original_id = int(os.path.splitext(image_file)[0])
                        new_id = nid
                        img_path = os.path.join(images_folder_path, image_file)
                        img = Image.open(img_path)

                        new_img_path = os.path.join(
                            val_folder, f"{new_id}_{original_id:03d}.jpg"
                        )
                        save_image(img, new_img_path)


def main():
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder")
    parser.add_argument("output_folder", type=str, help="Path to the output folder")
    # parser.add_argument('id_file_path', type=str, help='Path to the ID file')
    # parser.add_argument('num_ids', type=int, help='Number of unique IDs')

    args = parser.parse_args()

    # id_map = create_id_map(args.id_file_path, args.num_ids)
    process_images(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
