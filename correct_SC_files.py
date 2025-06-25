"""
This script moves files in a experiment directory into the correct subdirectories based on the new file structure convention.
"""

from pathlib import Path, PurePath
import shutil

root_dir = Path(r"/home/mawa/nas_a/Marvin/toy_example/Phase_0")
root_name = "alldata"

files_to_me_moved = [
    f"{root_name}.basis.hdf5",
    f"{root_name}.clusters.hdf5",
    f"{root_name}.clusters-merged.hdf5",
    f"{root_name}.log",
    f"{root_name}.overlap.hdf5",
    f"{root_name}.result.hdf5",
    f"{root_name}.result-merged.hdf5",
    f"{root_name}.templates.hdf5",
    f"{root_name}.templates-merged.hdf5",
]

# All files need to be moved into root_dir / "alldata"
for file_name in files_to_me_moved:
    source_file = root_dir / file_name
    if not source_file.exists():
        print(f"File {source_file} does not exist, skipping.")
        continue

    target_dir = root_dir / "alldata"
    target_file = target_dir / file_name
    if target_file.exists():
        print(f"Target file {target_file} already exists, skipping.")
        continue

    # Move file
    shutil.move(source_file, target_dir)

# %% Check text file and correct file locations
text_file = root_dir / "alldata.params"

if not text_file.exists():
    print(f"Text file {text_file} does not exist, skipping.")
else:
    with open(text_file, "r") as file:
        lines = file.readlines()

    # Process the lines to update paths
    updated_lines = []
    for line in lines:
        if "mapping" in line and "=" in line:
            # Extract the mapping path and keep only the filename
            parts = line.split("=")
            if len(parts) >= 2:
                path = parts[1].strip()
                # Handle potential Windows paths by removing comments and extracting filename safely
                clean_path = path.split("#")[0].strip()

                # %%
                if "\\" in clean_path and clean_path.count("\\") > 1:
                    # Convert to pathlib.Path object
                    filename = Path(clean_path.split("\\")[-1])
                else:
                    filename = PurePath(clean_path).name
                updated_lines.append(
                    f"mapping        = {filename}# Mapping of the electrode (see http://spyking-circus.rtfd.org)\n"
                )
            else:
                updated_lines.append(line)
        elif "output_dir" in line and "=" in line:
            # Clear the output_dir value
            parts = line.split("=")
            if len(parts) >= 2:
                updated_lines.append(
                    f"output_dir     =         # By default, generated data are in the same folder as the data.\n"
                )
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    # Write the updated content back to the file
    with open(text_file, "w") as file:
        file.writelines(updated_lines)
