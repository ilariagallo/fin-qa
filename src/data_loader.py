import json


class DataLoader:

    def __init__(self, filepath):
        self.data = self.load_data(filepath)
        self.pre_processing()

    def load_data(self, filepath):
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        return json_data

    def pre_processing(self):
        # Rename 'qa' to 'qa_0' if it exists to standardise all samples in the dataset
        for item in self.data:
            if 'qa' in item:
                item['qa_0'] = item.pop('qa')

