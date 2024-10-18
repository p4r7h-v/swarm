""""
This module provides a function to copy the directory structure and file contents of a directory to the clipboard.
"""
import os
import pyperclip

EXCLUDE_DIRS = {'testing', 'pycache', 'venv', 'git', 'env'}

def should_exclude(dirpath):
    """
    Determine if a directory should be excluded based on its name.
    """
    basename = os.path.basename(dirpath).lower()
    return any(excluded in basename for excluded in EXCLUDE_DIRS)

def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir.
    Ignores directories that contain certain names.
    """
    structure = {}
    for dirpath, dirnames, filenames in os.walk(rootdir):
        # Filter out unwanted directories
        dirnames[:] = [d for d in dirnames if not should_exclude(os.path.join(dirpath, d))]

        folder = os.path.relpath(dirpath, rootdir)
        subdir = structure
        if folder != '.':
            for part in folder.split(os.sep):
                subdir = subdir.setdefault(part, {})
        subdir.update({filename: None for filename in filenames})
    return structure

def format_structure(structure, indent=0):
    """
    Formats the directory structure into a string with the desired hierarchy format.
    """
    result = ''
    for key, value in structure.items():
        result += ' ' * indent + f'├── {key}\n'
        if isinstance(value, dict):
            result += format_structure(value, indent + 4)
    return result

def concatenate_files(rootdir, structure):
    """
    Concatenates the content of all files in the directory following the hierarchy structure.
    """
    result = ''
    for key, value in structure.items():
        if value is None:  # it's a file
            filepath = os.path.join(rootdir, key)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                result += f'{key}:\n{content}\n'
            except (UnicodeDecodeError, IsADirectoryError) as e:
                print(f"Could not read file {filepath} due to {e.__class__.__name__}: {e}")
        else:  # it's a directory
            subdir = os.path.join(rootdir, key)
            result += concatenate_files(subdir, value)
    return result

def get_clipboard(rootdir):
    # Generate the directory structure
    structure = get_directory_structure(rootdir)

    # Format the structure to a string
    structure_str = format_structure(structure)

    # Concatenate all file contents
    file_contents = concatenate_files(rootdir, structure)

    # Combine both parts
    result = structure_str + '\n' + file_contents

    # Copy the result to the clipboard
    pyperclip.copy(result)

    # Count words
    word_count = len(result.split())
    print("The directory structure and file contents have been copied to the clipboard.")
    print(f"The copied text contains {word_count} words.")
    return result