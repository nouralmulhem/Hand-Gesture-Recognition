import os

# specify the directory path containing the images
dir_path = './test_101/'

# get a list of all the files in the directory
files = os.listdir(dir_path)

# loop through the files and rename the images with ascending numbers
for i, file in enumerate(files):
    if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'): # check if the file is an image
        # construct the new file name with ascending numbers
        new_name = f'{i+1}.jpg' # change the extension to match the original image format

        # construct the full paths for the old and new file names
        old_path = os.path.join(dir_path, file)
        new_path = os.path.join(dir_path, new_name)

        # rename the image file
        os.rename(old_path, new_path)