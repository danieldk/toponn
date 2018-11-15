from os import path

class DefaultConfig:
    init_scale = 0.05
    hidden_size = 50
    keep_prob = 0.85
    keep_prob_input = 0.80
    max_epoch = 100

def path_relative_to_conf(conf_path, file_path):
    if path.isabs(file_path):
        return path

    return "%s/%s" % (path.dirname(path.abspath(conf_path)), file_path)

