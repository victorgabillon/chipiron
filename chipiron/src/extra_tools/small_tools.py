import os
import yaml


def mkdir(folder_path):
    try:
        os.mkdir(folder_path)
    except OSError as error:
        print(error)
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
        string += ' {:>30} : {}\n'.format(key, value)
    return string


def unique_int_from_list(a_list):
    # only coded for a list of 2 atm
    assert (len(a_list) == 2)
    x = a_list[0]
    y = a_list[1]
    return int(.5 * (x + y) * (x + y + 1) + y)  # Cantor pairing function
