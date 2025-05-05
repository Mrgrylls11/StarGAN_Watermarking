# --- START OF FILE data_loader.py ---

from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np # Needed for split

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, test_split_ratio=0.1): # Add split ratio
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs if selected_attrs else [] # Handle None case
        self.transform = transform
        self.mode = mode
        self.test_split_ratio = test_split_ratio
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        # Assign dataset based on mode AFTER preprocessing
        self.dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        self.num_images = len(self.dataset)


        # Fail early if dataset is empty for the current mode
        if self.num_images == 0:
            # Provide more specific feedback
            if len(self.train_dataset) + len(self.test_dataset) == 0:
                 msg = f"No valid entries found in attribute file or corresponding images missing. Please check image_dir '{self.image_dir}' and attr_path '{self.attr_path}'."
            elif self.mode == 'train':
                 msg = f"No data available for mode='train'. Check test_split_ratio ({self.test_split_ratio}) or data source."
            else: # mode == 'test'
                 msg = f"No data available for mode='test'. Check test_split_ratio ({self.test_split_ratio}) or data source."
            raise RuntimeError(msg)


    def preprocess(self):
        """Preprocess the CelebA attribute file and split into train/test."""
        try:
            lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        except FileNotFoundError:
            print(f"❌ Error: Attribute file not found at {self.attr_path}")
            # Raise error to stop execution if attr file is missing
            raise FileNotFoundError(f"Attribute file not found: {self.attr_path}")


        if len(lines) < 3:
             raise ValueError(f"Attribute file '{self.attr_path}' seems malformed or empty (less than 3 lines).")

        num_total_attrs_header = int(lines[0].strip()) # First line is the count
        all_attr_names = lines[1].split()

        if len(all_attr_names) != num_total_attrs_header:
              print(f"Warning: Number of attributes in header ({num_total_attrs_header}) does not match number of names found ({len(all_attr_names)}) in {self.attr_path}")


        print(f"Total available attributes: {len(all_attr_names)}")
        print(f"Selected attributes: {self.selected_attrs}")

        # Check if selected attributes are valid
        invalid_attrs = [attr for attr in self.selected_attrs if attr not in all_attr_names]
        if invalid_attrs:
             print(f"❌ Error: The following selected attributes are not found in the attribute file: {invalid_attrs}")
             print(f"Available attributes are: {all_attr_names}")
             raise ValueError("Invalid selected attributes provided.")


        # Build attribute mappings
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]  # Skip header lines (count and names)
        random.seed(1234) # Use a fixed seed for reproducible splits
        random.shuffle(lines)

        valid_entries = []
        missing_images_count = 0
        processed_lines = 0

        for i, line in enumerate(lines):
            processed_lines += 1
            split = line.split()
            if len(split) < 2: # Must have filename and at least one attribute value
                  print(f"Warning: Skipping malformed line {i+3} in {self.attr_path}: '{line}'")
                  continue

            filename = split[0]
            values = split[1:] # All remaining are attribute values (-1 or 1)

            # Check if image file exists
            image_path = os.path.join(self.image_dir, filename)
            if not os.path.exists(image_path):
                # Print only the first few missing images to avoid flooding logs
                if missing_images_count < 5:
                    print(f"Warning: Missing image file: {image_path}")
                elif missing_images_count == 5:
                    print("Warning: Further missing image messages will be suppressed.")
                missing_images_count += 1
                continue # Skip this entry if image is missing

            # Check if the number of values matches the expected number of attributes
            if len(values) != len(all_attr_names):
                 print(f"Warning: Skipping line {i+3} for {filename}. Expected {len(all_attr_names)} attribute values, found {len(values)}.")
                 continue


            # Extract labels for selected attributes
            label = []
            valid_entry = True
            for attr_name in self.selected_attrs:
                try:
                    idx = self.attr2idx[attr_name]
                    # Convert '-1'/'1' strings to 0/1 boolean/float representation
                    label.append(values[idx] == '1') # True if '1', False if '-1' or other
                except IndexError:
                     # This should theoretically not happen if checks above passed
                     print(f"Error: Index out of bounds for attribute '{attr_name}' on line {i+3}. Skipping entry.")
                     valid_entry = False
                     break
                except KeyError:
                     # This should not happen due to the check at the beginning
                     print(f"Error: Attribute '{attr_name}' not in attr2idx mapping. Skipping entry.")
                     valid_entry = False
                     break

            if valid_entry:
                valid_entries.append([filename, label])

        if missing_images_count > 0:
             print(f"Total missing image files: {missing_images_count}")

        num_valid = len(valid_entries)
        if num_valid == 0:
            print("❌ Error: No valid entries found after preprocessing. Check image paths and attribute file format.")
            # No need to raise here, the __init__ check handles empty dataset
            return

        # Perform train/test split
        split_index = int(self.test_split_ratio * num_valid)
        if split_index == 0 and self.test_split_ratio > 0 and num_valid > 0:
             split_index = 1 # Ensure at least one test sample if ratio > 0 and possible
        if split_index >= num_valid: # Ensure at least one training sample if possible
             split_index = max(0, num_valid - 1)


        self.test_dataset = valid_entries[:split_index]
        self.train_dataset = valid_entries[split_index:]


        print(f'✅ Finished preprocessing the CelebA dataset. Processed lines: {processed_lines}. Valid entries: {num_valid}.')
        print(f"Train samples: {len(self.train_dataset)} | Test samples: {len(self.test_dataset)}")


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        # Choose the correct dataset based on the instance's mode
        dataset = self.dataset # Use the dataset assigned in __init__
        if index >= len(dataset):
             raise IndexError(f"Index {index} out of bounds for {self.mode} dataset with size {len(dataset)}")

        filename, label = dataset[index]
        image_path = os.path.join(self.image_dir, filename)

        try:
            # Use PIL to open image and convert to RGB
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
             # Handle cases where image might have been deleted between preprocess and getitem
             print(f"❌ Error: Image file not found during __getitem__: {image_path}")
             # Return a dummy tensor or raise error, depending on desired behavior
             # Returning dummy data might hide issues. Raising is safer.
             raise FileNotFoundError(f"Image file missing for entry {index}: {image_path}")
        except Exception as e:
             print(f"❌ Error opening or converting image {image_path}: {e}")
             raise IOError(f"Failed to process image for entry {index}: {image_path}")


        # Apply transformations and convert label to FloatTensor
        return self.transform(image), torch.FloatTensor(label)


    def __len__(self):
        """Return the number of images in the selected mode's dataset."""
        return self.num_images


# ✅ This function is now correctly outside the class
def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128,
                  batch_size=16, dataset='CelebA', mode='train', num_workers=1):
        """Build and return a data loader."""
        transform = []
        if mode == 'train':
            transform.append(T.RandomHorizontalFlip())
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR)) # Specify interpolation
        transform.append(T.ToTensor()) # Converts [0, 255] -> [0, 1]
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))) # Normalizes [0, 1] -> [-1, 1]
        transform = T.Compose(transform)

        if dataset == 'CelebA':
             # Pass the mode to the CelebA dataset constructor
            dataset_obj = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
        elif dataset == 'RaFD':
             # ImageFolder assumes structure like image_dir/class_name/image.jpg
            if not os.path.isdir(image_dir):
                 raise FileNotFoundError(f"RaFD image directory not found or not a directory: {image_dir}")
            try:
                dataset_obj = ImageFolder(image_dir, transform)
                if len(dataset_obj) == 0:
                     print(f"Warning: ImageFolder found no images in {image_dir}. Check directory structure.")
            except Exception as e:
                 print(f"❌ Error initializing ImageFolder for RaFD at {image_dir}: {e}")
                 raise # Re-raise the exception

        else:
            raise ValueError(f"Unsupported dataset type: {dataset}")

        # Check again if dataset_obj is empty after potential warnings
        if len(dataset_obj) == 0:
             print(f"Error: No data loaded for dataset '{dataset}' in mode '{mode}'. Dataloader cannot be created.")
             # Returning None or an empty loader might be better than erroring here,
             # depends on how the main script handles it. Let's return None.
             return None

        data_loader = data.DataLoader(dataset=dataset_obj,
                                      batch_size=batch_size,
                                      shuffle=(mode == 'train'),
                                      num_workers=num_workers,
                                      pin_memory=True, # Usually helps speed up CPU->GPU transfer
                                      drop_last=True) # Drop last incomplete batch, common in GAN training
        return data_loader

# --- END OF FILE data_loader.py ---