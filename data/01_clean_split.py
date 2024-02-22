from collections import Counter
import random
from pathlib import Path


# Load all files path
data_dir = Path('./data/01_raw/')

file_names = []
file_paths = []

# Loop over the image files in the data directory and its subdirectories

for image_file in data_dir.glob('**/*.jpg'):
    # Get the path to the image file
    image_path = image_file.as_posix()
    index = image_path.rfind('/')
    file_name = image_path[index+1:]
    file_paths.append(image_path)
    file_names.append(file_name)
    
# find no repeated image files
file_name_counter = Counter(file_names)
non_repeated_file_names = [name for name, count in file_name_counter.items() if count == 1]

# remove them from image paths
non_repeated_paths = [path for path in file_paths if path[path.rfind('/')+1:] in non_repeated_file_names]

# sample each subfolder 
num_test_images=50
subfolders = ["crescentes", "hipercellularity", "membranous", "normal", "Podocitopatia_stain", "sclerosis"]
for folder in subfolders:
    filtered_paths = [path for path in non_repeated_paths if folder in path ]
    test_image_paths = random.sample(filtered_paths, num_test_images)
    train_image_paths = [path for path in filtered_paths if path not in test_image_paths]

# save in data split folder
    for f in test_image_paths:
        dst = Path(f.replace("data/01_raw/","data/02_data_split/test_data/"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(f).read_bytes())
        

    for f in train_image_paths:
        dst = Path(f.replace("data/01_raw/","data/02_data_split/train_data/"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(f).read_bytes())