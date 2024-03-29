import math
import random


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
        ordered_list_elements,
        random_generator
):
    length_list = len(ordered_list_elements)
    assert (length_list > 0)
    weights = [1 / (index + 1) / (math.log(math.e * (index + 1))) ** 0 for index in range(length_list)]
    # print('44',weights, ordered_list_elements)
    picked_element = random_generator.choices(ordered_list_elements, weights=weights, k=1)
    return picked_element[0]


def zipf_picks_random_bool(ordered_list_elements, bool_func, random_generator):
    length_list = len(ordered_list_elements)
    assert (length_list > 0)
    weights = [float(bool_func(index)) / (index + 1) / (math.log(math.e * (index + 1))) ** 2 for index in
               range(length_list)]

    picked_element = random_generator.choices(list(range(length_list)), weights=weights, k=1)
    return picked_element[0]


if __name__ == '__main__':
    import random

    randomg = random.Random()
    r = {1: 2., 2: 3, 3: 4}
    res = zipf_picks(
        random_generator=randomg,
        ranks_values=r,
        random_pick=False,
        shift=False
    )
    print(res)
