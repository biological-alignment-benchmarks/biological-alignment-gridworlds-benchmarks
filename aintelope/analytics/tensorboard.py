import os
import subprocess
from collections import defaultdict
import tempfile


def get_common_subdirs(base_dirs):
    # Get all subdirectories for each selected directory
    subdir_counts = defaultdict(int)
    for dir_path in base_dirs:
        try:
            subdirs = [
                d
                for d in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, d))
            ]
            for subdir in subdirs:
                subdir_counts[subdir] += 1
        except Exception as e:
            print(f"Error reading directory {dir_path}: {e}")

    # Find subdirectories that exist in all selected directories
    common_subdirs = [
        subdir for subdir, count in subdir_counts.items() if count == len(base_dirs)
    ]

    return sorted(common_subdirs)


def get_homedir():
    # hacky way to get the correct path when calling from anywhere.
    directory = os.path.dirname(os.path.abspath(__file__))
    directory += "/../../outputs/"
    return os.path.normpath(directory)


def user_choice(directories):
    # Print available directories
    print("\nAvailable directories:")
    for idx, directory in enumerate(directories, 1):
        print(f"{idx}. {directory}")

    # Get user input for first level
    print(
        "\nEnter the numbers of directories you want to compare (separated by spaces):"
    )
    selected = input("> ").strip().split()

    # Get full paths of selected directories
    selected_dirs = []
    for num in selected:
        try:
            idx = int(num) - 1
            dir_name = directories[idx]
            selected_dirs.append(dir_name)
        except ValueError:
            print(f"Invalid input: {num}")
            return
    return selected_dirs


def dir_collection():
    base_dir = get_homedir()
    print("Looking from " + base_dir)

    # Get all experiment directories
    directories = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    selected_dirs = user_choice(directories)

    selected_dirs = [base_dir + "/" + dir for dir in selected_dirs]

    # Find common subdirectories
    common_subdirs = get_common_subdirs(selected_dirs)
    selected_subdirs = user_choice(common_subdirs)

    # Accumulate all the dirs to be sent forward, label : directory
    logdirs = {}
    for exp_dir in selected_dirs:
        for sub_dir in selected_subdirs:
            label = exp_dir.split("/")[-1] + "_" + sub_dir
            full_path = exp_dir + "/" + sub_dir + "/tensorboard/"
            logdirs[label] = full_path

    return logdirs


def create_tensorboard_view(paths):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="tensorboard_view_")
    print(f"Created temporary directory: {temp_dir}")

    # Create symlinks in the temporary directory
    for name, path in paths.items():
        full_path = os.path.abspath(os.path.expanduser(path))
        link_path = os.path.join(temp_dir, name)
        try:
            os.symlink(full_path, link_path)
            print(f"Created symlink: {name} -> {full_path}")
        except Exception as e:
            print(f"Error creating symlink for {name}: {e}")
            return

    # Launch tensorboard with the temp directory
    print("\nLaunching TensorBoard...")
    subprocess.run(["tensorboard", "--logdir", temp_dir])


if __name__ == "__main__":
    dirs = dir_collection()
    create_tensorboard_view(dirs)
