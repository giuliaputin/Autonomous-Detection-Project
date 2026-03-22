import shutil
import os

base_path = r"c:\Users\unigi\Desktop\CFWorkshop\Autonomous Detection Project"

# Copy data directory
print("Step 1: Moving data directory...")
src_data = os.path.join(base_path, "data")
dst_data = os.path.join(base_path, "qr_detection", "data")

if os.path.exists(src_data):
    for item in os.listdir(src_data):
        src_item = os.path.join(src_data, item)
        dst_item = os.path.join(dst_data, item)
        if os.path.isdir(src_item):
            if os.path.exists(dst_item):
                shutil.rmtree(dst_item)
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)
        print(f"  Moved: {item}")

# Delete duplicate Python files
print("\nStep 2: Deleting duplicate Python files from qr_detection root...")
files_to_delete = ["cli.py", "detector.py", "gui.py", "image_processor.py", "interface.py", "main.py"]
for f in files_to_delete:
    path = os.path.join(base_path, "qr_detection", f)
    if os.path.exists(path):
        os.remove(path)
        print(f"  Deleted: {f}")
    else:
        print(f"  Not found: {f}")

# Delete root data folder
print("\nStep 3: Deleting root-level data/ folder...")
if os.path.exists(src_data):
    shutil.rmtree(src_data)
    print(f"  Deleted: data/")
else:
    print(f"  Already deleted")

print("\n✓ Reorganization complete!")
