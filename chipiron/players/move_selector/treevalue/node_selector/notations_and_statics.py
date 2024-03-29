import math
import random

from typing import TypeVar

T = TypeVar('T')


def zipf_picks(
        ranks_values: dict[int, int | float],
        random_generator: random.Random,
        shift: bool = False,
        random_pick: bool = False
) -> int:
    """ this function returns the min of the index times the rank times log terms"""
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
        weight: float = value * shifted_rank * log_term + .0001
        weights.append(weight)
        #  print('value', weight, value, shifted_rank, log_term)
        #  print('wegith', best_weight, weight, best_weight, best_rank)

        if best_weight is None or weight < best_weight:
            best_weight = weight
            best_rank = rank
    # print('bestrank', best_rank)

    if random_pick:
        raise Exception('nodt coded yet')  # not codeed properly yet
        # choices = random_generator.choices(list(ranks.keys()), weights=weights, k=1)
    # print('bestrankzz')

    # return choices[0]
    else:
        # print('bestrank', weights)

        return best_rank


def zipf_picks_random(
        ordered_list_elements: list[T],
        random_generator: random.Random
) -> T:
    length_list = len(ordered_list_elements)
    assert (length_list > 0)
    weights = [1 / (index + 1) / (math.log(math.e * (index + 1))) ** 0 for index in range(length_list)]
    # print('44',weights, ordered_list_elements)
    picked_element = random_generator.choices(ordered_list_elements, weights=weights, k=1)
    return picked_element[0]
