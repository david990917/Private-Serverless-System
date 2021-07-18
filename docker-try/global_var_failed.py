global _global_dict
_global_dict = {}


def _setValue(key, value):
    _global_dict[key] = value


def _getValue(key):
    return _global_dict[key]
