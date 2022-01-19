import importlib


def import_obj(path):
    obj_name = path.split('.')[-1]
    module_path = '.'.join([i for i in path.split('.')][:-1])
    module = importlib.import_module(module_path)
    obj = getattr(module, obj_name)
    return obj
