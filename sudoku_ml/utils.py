import importlib


def import_obj(path):
    obj_name = path.split('.')[-1]
    module_path = '.'.join([i for i in path.split('.')][:-1])
    module = importlib.import_module(module_path)
    obj = getattr(module, obj_name)
    return obj


def parse_remove(value):
    removed = [int(i) for i in value.split(',')][:2]
    removed = (removed + [removed[0]+1]) if len(removed) == 1 else removed
    return removed
