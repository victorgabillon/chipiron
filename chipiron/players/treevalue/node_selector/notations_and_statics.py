import math
import numpy as np


def zipf_picks(list_elements, value_of_element):
    length_list = len(list_elements)
    assert (length_list > 0)
    best_index = 0
    best_value = value_of_element(0)
    values = []
    for index in range(length_list):
        # print('%va', value_of_element(index) , (index + 1) * (math.log(math.e * (index + 1)))**2)
        value = value_of_element(index) * (index + 1) * (math.log(math.e * (index + 1))) ** 2
        if value < best_value:
            best_value = value
            best_index = index
        values.append(value)
    return best_index, best_value


def zipf_picks_random(ordered_list_elements, random_generator):
    length_list = len(ordered_list_elements)
    assert (length_list > 0)
    weights = [1 / (index + 1) / (math.log(math.e * (index + 1))) ** 0 for index in range(length_list)]
    # print('44',weights)
    picked_element = random_generator.choices(ordered_list_elements, weights=weights, k=1)
    return picked_element[0]


def zipf_picks_random_bool(ordered_list_elements, bool_func, random_generator):
    length_list = len(ordered_list_elements)
    assert (length_list > 0)
    weights = [float(bool_func(index)) / (index + 1) / (math.log(math.e * (index + 1))) ** 2 for index in
               range(length_list)]

    picked_element = random_generator.choices(list(range(length_list)), weights=weights, k=1)
    return picked_element[0]


def zipf_picks_random_weird(list_elements, bool_func, random_generator):
    length_list = len(list_elements)
    assert (length_list > 0)
    weights = [0] * length_list
    bools = [bool_func(index) for index in range(length_list)]
    for index in range(length_list):
        if bool_func(index):
            weights[index] = 1 / (index + 1) / (math.log(math.e * (index + 1))) ** 0
            # best_index = [index] ##debug
            # print('@',weights,list(range(length_list)),bools)
    assert (bools[-1] == True)
    best_index = random_generator.choices(list(range(length_list)), weights=weights, k=1)

    return best_index[0], None


