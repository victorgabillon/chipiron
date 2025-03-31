"""
This module contains functions for selecting elements based on Zipf distribution.

Zipf distribution is a discrete probability distribution that models the occurrence of elements in a dataset.
The functions in this module provide methods for selecting elements based on their rank and value.

"""

import math
import random


def zipf_picks(
    ranks_values: dict[int, int | float],
    random_generator: random.Random,
    shift: bool = False,
    random_pick: bool = False,
) -> int:
    """
    Selects an element based on its rank and value.

    Args:
        ranks_values (dict[int, int | float]): A dictionary containing the ranks and values of the elements.
        random_generator (random.Random): A random number generator.
        shift (bool, optional): Whether to shift the ranks. Defaults to False.
        random_pick (bool, optional): Whether to perform a random pick. Defaults to False.

    Returns:
        int: The rank of the selected element.

    Raises:
        Exception: If random_pick is True (not implemented yet).

    """
    shift_rank: int

    if shift:
        shift_rank = min(ranks_values.keys())
    else:
        shift_rank = 0

    best_weight: float | None = None
    best_rank: int = shift_rank

    weights = []
    for rank, value in ranks_values.items():

        shifted_rank = rank - shift_rank + 1
        log_term: float = (math.log(math.e * shifted_rank)) ** 2
        weight: float = value * shifted_rank * log_term + 0.0001
        weights.append(weight)

        if best_weight is None or weight < best_weight:
            best_weight = weight
            best_rank = rank

    if random_pick:
        raise Exception("nodt coded yet")  # not codeed properly yet
        # choices = random_generator.choices(list(ranks.keys()), weights=weights, k=1)

    else:
        return best_rank


def zipf_picks_random[T](
    ordered_list_elements: list[T], random_generator: random.Random
) -> T:
    """
    Selects a random element from an ordered list based on Zipf distribution.

    Args:
        ordered_list_elements (list[T]): A list of elements.
        random_generator (random.Random): A random number generator.

    Returns:
        T: The selected element.

    Raises:
        AssertionError: If the length of the list is 0.

    """
    length_list = len(ordered_list_elements)
    assert length_list > 0
    weights = [
        1 / (index + 1) / (math.log(math.e * (index + 1))) ** 0
        for index in range(length_list)
    ]
    picked_element = random_generator.choices(
        ordered_list_elements, weights=weights, k=1
    )
    return picked_element[0]
