import yaml
import os

class Config(dict):
    def __init__(self, cfg_path = None, **kwargs):
        cfg = {}
        if cfg_path is not None and os.path.exists(cfg_path):
            cfg = self.load_cfg(cfg_path)  

        super(Config, self).__init__(**kwargs)

        cfg.update(self)
        self.update(cfg)
        self.cfg = cfg

    def load_cfg(self, load_path):
        with open(load_path, "r") as fi:
            try:
                cfg = yaml.safe_load(fi)
            except yaml.YAMLError as exc:
                print(exc)
        return cfg if cfg is not None else {}

    def write_cfg(self, write_path=None):
        if write_path is None:
            write_path = './default_config.yaml'

        dump = {k:v for k,v in self.items() if k != 'cfg'}
        with open(write_path, "w") as fw:
            yaml.safe_dump(dump, fw, default_flow_style=False)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

if __name__ == '__main__':

    cfgaa = {
        'first' : 1,
        'second' : 2,
        'third' : True,
        'fouth' : 'Fourth',
    } 

    cfgdd = Config(**cfgaa)
    cfgdd.write_cfg()