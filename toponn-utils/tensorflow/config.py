from os import path

class DefaultConfig:
    batch_size = 128
    init_scale = 0.05
    hidden_size = 50
    hidden_dropout = 0.85
    input_dropout = 0.80
    max_epoch = 100

def path_relative_to_conf(conf_path, file_path):
    if path.isabs(file_path):
        return path

    return "%s/%s" % (path.dirname(path.abspath(conf_path)), file_path)

