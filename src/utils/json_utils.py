import json

'''
1. load json file
2. update json file
3. save json file
'''

def load_json(file_path: str) -> dict:
    """
    Load a JSON file and return its content as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Content of the JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def update_json(file_path: str, key: str, value: str) -> None:
    """
    Update a JSON file by adding or updating a key-value pair.

    Args:
        file_path (str): Path to the JSON file.
        key (str): Key to be added or updated.
        value (str): Value to be associated with the key.
    """
    data = load_json(file_path)
    data[key] = value
    save_json(file_path, data)


def save_json(file_path: str, data: dict) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        data (dict): Dictionary to be saved.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)