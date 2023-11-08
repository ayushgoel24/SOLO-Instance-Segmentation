import yaml
from torchvision import transforms

class ConfigurationUtils:

    @staticmethod
    def load_config_from_yaml(config_path): 
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    @staticmethod
    def create_transforms_from_config(config):
        composed_transform = []
        for transform_config in config:
            for transform_class, params in transform_config.items():
                transform_class = eval(transform_class)  # Convert the class name from string to class
                composed_transform.append(transform_class(**params))
        return transforms.Compose(composed_transform)
    
    @staticmethod
    def load_transform(key):
        config = ConfigurationUtils.load_config_from_yaml('config/data_config.yaml')
        return ConfigurationUtils.create_transforms_from_config(config[key])