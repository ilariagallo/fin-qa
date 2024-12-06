import json


class DataLoader:
    """Data Loader class to load the data from json"""

    def __init__(self, filepath: str):
        """
        Constructor for data loader

        :param filepath: path to the dataset
        """
        self.data = self.load_data(filepath)
        self.pre_processing()

    def load_data(self, filepath):
        """
        Loads data from json

        :param filepath: path to the dataset
        """
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        return json_data

    def pre_processing(self):
        """
        Data pre-processing and clean up
        """
        # Rename 'qa' to 'qa_0' if it exists to standardise all samples in the dataset
        for item in self.data:
            if 'qa' in item:
                item['qa_0'] = item.pop('qa')

