import os
import yaml
import json
import logging
import joblib
from typing import Any
from box import ConfigBox
from ensure import ensure_annotations
from src.utils.exception import CustomException


@ensure_annotations
def read_yaml(path_to_yaml: str) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox object (dot-accessible dict).

    Args:
        path_to_yaml (str): Path to YAML file

    Returns:
        ConfigBox: Parsed YAML content
    """
    try:
        if not os.path.exists(path_to_yaml):
            raise FileNotFoundError(f"YAML file not found at: {path_to_yaml}")

        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)

        logging.info(f"YAML file loaded successfully from: {path_to_yaml}")

        return ConfigBox(content)

    except Exception as e:
        raise CustomException(f"Error reading YAML file: {e}")


@ensure_annotations
def create_directories(paths: list, verbose: bool = True):
    """
    Create multiple directories.

    Args:
        paths (list): List of directory paths
        verbose (bool): Log creation info
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: str, data: dict):
    """
    Save dictionary as JSON file.
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"JSON saved at: {path}")
    except Exception as e:
        raise CustomException(e)


@ensure_annotations
def load_json(path: str) -> dict:
    """
    Load JSON file into dictionary.
    """
    try:
        with open(path, "r") as f:
            content = json.load(f)
        logging.info(f"JSON loaded from: {path}")
        return content
    except Exception as e:
        raise CustomException(e)


@ensure_annotations
def save_bin(data: Any, path: str):
    """
    Save binary file (model, preprocessor, etc.)
    """
    try:
        joblib.dump(data, path)
        logging.info(f"Binary file saved at: {path}")
    except Exception as e:
        raise CustomException(e)


@ensure_annotations
def load_bin(path: str) -> Any:
    """
    Load binary file.
    """
    try:
        data = joblib.load(path)
        logging.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        raise CustomException(e)


def load_object(path: str) -> Any:
    """
    Load a Python object from a binary file.
    """
    try:
        obj = joblib.load(path)
        if obj is None:
            raise ValueError(f"Loaded object is None from: {path}")
        logging.info(f"Object loaded from: {path}")
        return obj
    except Exception as e:
        raise CustomException(e)