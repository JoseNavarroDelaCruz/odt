import os
import numpy as np
import pandas as pd 

# Define the source directory
if os.getenv("SLURM_JOB_ID"):
    # HPC environment
    data_dir = os.path.join(os.environ["HOME_REPO"], "odt", "data")
else:
    # Local machine (customize for your actual path)
    data_dir = r"C:\Users\navarrodelacruz\Documents\GitHub\odt\data"

output_dir = data_dir


# Debugging: Confirm directory exists
if not os.path.exists(data_dir):
    print(f"Error: The directory {data_dir} does not exist!")
    exit()

# Define parameters
num_iterations = 5000  # Number of variations per file
noise_std_dev = 0.05   # Standard deviation for Gaussian noise
removal_fraction = 0.02  # Fraction of rows to remove per iteration

# Choose which files to include (by substring match)
filter_names = ["glass"]  # <-- Edit this list to match desired datasets

# Get list of files in the data directory that match one of the substrings
all_files = [
    f for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f)) and any(name in f for name in filter_names)
]

# Debugging: Print found files
if not all_files:
    print("No files found in the directory!")
else:
    print(f"Found {len(all_files)} file(s): {all_files}")

# Process each file in the data directory
for filename in all_files:
    file_path = os.path.join(data_dir, filename)
    
    try:
        # Try reading the file as a CSV/TXT
        df = pd.read_csv(file_path, header=None, delimiter=None, engine='python')

        print(f"\nProcessing file: {filename}")
        print(df.head())

        # Identify the number of columns (assuming last one is the label)
        num_features = df.shape[1] - 1

        # Extract base name (e.g., "glass" from "glass.txt")
        base_name = filename.split('.')[0]

        # === Create 'augmented_datasets' directory if not exist ===
        project_root = os.path.abspath(os.path.join(data_dir, ".."))  # one level above /data
        augmented_root = os.path.join(project_root, "augmented_datasets")
        os.makedirs(augmented_root, exist_ok=True)

        # === Create subdirectory for this dataset ===
        dataset_dir = os.path.join(augmented_root, base_name)
        os.makedirs(dataset_dir, exist_ok=True)

        for i in range(1, num_iterations + 1):
            df_aug = df.copy()

            # Apply Gaussian noise to feature columns
            noise = np.random.normal(0, noise_std_dev, df_aug.iloc[:, :-1].shape)
            df_aug.iloc[:, :-1] += noise

            # Randomly remove a fraction of rows
            num_rows_to_remove = int(len(df_aug) * removal_fraction)
            if num_rows_to_remove > 0:
                rows_to_remove = np.random.choice(df_aug.index, num_rows_to_remove, replace=False)
                df_aug = df_aug.drop(rows_to_remove)

            # Save to augmented_datasets/<base_name>/<base_name>_i.ext
            file_ext = filename.split('.')[-1]
            new_filename = f"{base_name}_{i}.{file_ext}"
            output_path = os.path.join(dataset_dir, new_filename)
            df_aug.to_csv(output_path, index=False, header=False)

            if i % 100 == 0 or i == num_iterations:
                print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("\nData augmentation complete!")

