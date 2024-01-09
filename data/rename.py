# import glob
# import os

# images = glob.glob("./dataset/*")

# for idx,image in enumerate(images):
#     old_file_path = os.path.abspath(image)
#     new_file_path = os.path.dirname(old_file_path)
#     new_file_path = os.path.abspath(f"{new_file_path}/{idx}.jpg")
#     os.rename(old_file_path, new_file_path)


import glob
import os

images = glob.glob("./dataset/*")

for idx, image in enumerate(images):
    print(image)
    # Get the base name of the image (excluding the path)
    base_name = os.path.basename(image)

    # Get the directory of the image
    image_directory = os.path.dirname(os.path.abspath(image))

    # Create a new unique file name using the original name and index
    new_file_name = f"{idx}.jpg"

    # Construct the full path for the new file
    new_file_path = os.path.join(image_directory, new_file_name)

    # Rename the file
    print(new_file_path)
    print()

    os.rename(image, new_file_path)
