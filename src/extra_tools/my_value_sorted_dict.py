from bidict import bidict


class MyValueSortedDict:

    def __init__(self):
        self.dic = {}

    def __setitem__(self, key, value):
        self.dic[key] = value

    def __getitem__(self, key):
        return self.dic[key]

    def items(self):
        return self.dic.items()

    def __len__(self):
        return len(self.dic)

    def __iter__(self):
        return iter(self.dic)

    def sort_dic(self):
        self.dic = dict(sorted(self.dic.items(), key=lambda item: item[1]))
        # {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}


class MyValueSortedDict2:

    def __init__(self):
        self.rank_key = bidict()
        self.values = {}

    def __setitem__(self, key, value):
        if key in self.rank_key:
            # update the rank
            pass
        else:
            # insert and update
            length = len(self.rank_key)
            self.rank_key[key] = value
        sel.values[key] = value

    def __getitem__(self, rank):
        return self.rank_key[rank]

    def __iter__(self):
        assert (1 == 0)  # here to check how its used and how to code it
