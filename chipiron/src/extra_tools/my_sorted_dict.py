from bidict import bidict


class MySortedDict:

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

    def popitem(self):
        return self.dic.popitem()

    def sort_dic(self):
        self.dic = dict(sorted(self.dic.items(), key=lambda item: item[0]))
        # {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}


