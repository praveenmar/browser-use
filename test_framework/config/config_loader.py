import yaml

class ConfigLoader:
    @staticmethod
    def load_config(path: str) -> dict:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
