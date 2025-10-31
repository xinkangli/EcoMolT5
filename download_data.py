from datasets import load_dataset

# Download the dataset
ds = load_dataset("haitengzhao/molecule_property_instruction")

# Save it locally (optional: specify a path)
ds.save_to_disk("gimlet_data")
