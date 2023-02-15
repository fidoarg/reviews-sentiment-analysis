import tarfile
import os
import shutil
import pandas as pd

from typing import Tuple

def decompress_tarfile(source_path: str, dest_dir: str = None, new_folder_name: str = None) -> None:
    """
    Receives the source file, the destination folder where to save the decompressed data 
    and the folder name for the decompressed data.

    Parameters
    ----------
    source_path : str
        Location of the tarfile with the movie reviews.

    dest_dir : str (optional)
        Location where the decompressed data is going to be saved.
        If no dest_dir is provided,the destination folder will be the same as the source folder

    new_folder_name : str (optional)
        Name for the new folder to be saved. 
        If no name is provided,the folder name will be 'model_data'
    
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError("File parameter not found")

    source_dir = os.path.dirname(source_path)
    dest_dir = source_dir if dest_dir is None else dest_dir
    new_folder_name = 'model_data' if new_folder_name is None else new_folder_name

    with tarfile.open(source_path, 'r') as tar_file:
        tar_file.extractall(path= dest_dir)
    
    if os.path.isdir(os.path.join(dest_dir, new_folder_name)):
        shutil.rmtree(os.path.join(dest_dir, new_folder_name))

    contents = os.listdir(dest_dir)
    directories = [
        os.path.join(dest_dir, element)
        for element in contents
        if os.path.isdir(os.path.join(dest_dir, element))
    ]

    old_folder_name = directories[0]
    new_folder_name = os.path.join(dest_dir, 'model_data')

    os.rename(old_folder_name, new_folder_name)


def get_model_raw_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ...
    # Initialize empty lists to store review text and sentiment label
    reviews = []
    sentiments = []

    # Loop through each folder (positive and negative)
    for sentiment in ["pos", "neg"]:
        folder_path = os.path.join(data_dir, sentiment)
    
    # Loop through each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Read the contents of the file
            with open(file_path, "r") as f:
                review_text = f.read()
            
            # Append the review text and sentiment label to the lists
            reviews.append(review_text)
            sentiments.append(sentiment)
    
    sentiments = map(lambda x: x == 'pos', sentiments)

    # Create a pandas dataframes from the lists
    X, y = pd.DataFrame({"Review": reviews}), pd.DataFrame({"Sentiment": sentiments}).astype('int8')

    return (X, y)

