import math
from itertools import islice
import numpy as np
ZIPF_STYLE = [ZIPF, ZIPF_ONE, ZIPF_TWO] = range(3)


def zipf_picks(list_elements, value_of_element):
    length_list = len(list_elements)
    assert (length_list > 0)
    best_index = 0
    best_value = value_of_element(0)
    values = []
    for index in range(length_list):
        #print('%va', value_of_element(index) , (index + 1) * (math.log(math.e * (index + 1)))**2)
        value = value_of_element(index) * (index + 1) * (math.log(math.e * (index + 1)))**2
        if value < best_value:
            best_value = value
            best_index = index
        values.append(value)
    #print('%values',values)

    return best_index, best_value


def nth_key(dct, n):
    it = iter(dct)
    # Consume n elements.
    next(islice(it, n, n), None)
    # Return the value at the current position.
    # This raises StopIteration if n is beyond the limits.
    # Use next(it, None) to suppress that exception.
    return next(it)


def softmax(x,temperature):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))*temperature)
    return e_x / e_x.sum(axis=0) # only difference