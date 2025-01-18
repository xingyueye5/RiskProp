import os


def count_folders_with_name(root_dir, target_name):
    """
    Recursively count the number of folders with the specified name.

    Args:
        root_dir (str): The root directory to start the search.
        target_name (str): The name of the folders to look for.

    Returns:
        int: The count of folders with the specified name.
    """
    count = 0
    for dirpath, dirnames, _ in os.walk(root_dir):
        count += dirnames.count(target_name)
    return count


# Example usage
if __name__ == "__main__":
    root_directory = "data/MM-AU/CAP-DATA"  # Replace with your directory path
    folder_name = "000024"  # Replace with the folder name you want to search for
    result = count_folders_with_name(root_directory, folder_name)
    print(f"Number of folders named '{folder_name}': {result}")
