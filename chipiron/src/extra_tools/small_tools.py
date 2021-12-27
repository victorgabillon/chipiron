import os
import yaml


def mkdir(folder_path):
    print('edfr')
    try:
        os.mkdir(folder_path)
    except OSError:
        print("Creation of the directory %s failed" % folder_path)
    else:
        print("Successfully created the directory %s " % folder_path)


def yaml_fetch_args_in_file(path_file):
    with open(path_file, 'r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args


def dict_alphabetic_str(dic):
    string = ''
    for key, value in sorted(dic.items()):
        string += ' {:>30} : {}\n'.format(key,value)
    return string
