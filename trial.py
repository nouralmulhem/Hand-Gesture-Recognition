import os

# Path to the folder containing images

folder_path = os.getcwd()

# folder_path = 'D:/TempDesktop/second_term/NN/project/Hand-Gesture-Recognition'
# Path to the text file containing image filenames
txt_file_path = './try.txt'

# Read the image filenames from the text file
with open(os.path.join(folder_path, txt_file_path), 'r') as f:
    image_filenames = f.read().splitlines()

# Iterate over the image filenames
for filename in image_filenames:
    # Construct the full path to the image file
    image_path = os.path.join(folder_path, filename)

    # Check if the image file exists
    if os.path.isfile(image_path):
        # Delete the image file
        os.remove(image_path)
        print(f"Deleted {filename}")
    else:
        print(f"{filename} not found")
