
class ConfigLoader:
    def __init__(self, config_file):
        self.config = {}
        print("\033[93m-- Loading config file... --\033[0m")
        try:
            self._load_config(config_file)
            print("\033[92m-- Finished loading config successfully! --\033[0m")
        except FileNotFoundError:
            print(f"\033[91m-- Error: Config file '{config_file}' not found. --\033[0m")
        except Exception as e:
            print(f"\033[91m-- Error loading config file: {e} --\033[0m")
    
    def _load_config(self, config_file):
        with open(config_file, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    key, value = map(str.strip, line.split('=', 1))
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    else:
                        value = value.strip('"').strip("'")
                    self.config[key] = value
                except ValueError:
                    print(f"\033[91m-- Warning: Skipping invalid line in config: '{line}' --\033[0m")
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def print_config(self):
        print("\n\033[96m-- Configuration Parameters:\033[0m")
        print("\033[95m" + "=" * 40 + "\033[0m")
        for key, value in self.config.items():
            print(f"\033[93m--\033[0m \033[94m{key:20}\033[0m: \033[92m{value}\033[0m")
        print("\033[95m" + "=" * 40 + "\033[0m")
