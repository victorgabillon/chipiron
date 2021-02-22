class A:
    def __init__(self):
        super().__init__()
        print('init A')
        self.b = {}

    def x(self):
        print('A')


class B(A):
    def __init__(self):
        super().__init__()
        print('init B')
        self.b = {}

    def x(self):
        print('B')
        super().x()


class C(A):
    def __init__(self):
        super().__init__()
        print('init C')

        self.b = {}

    def x(self):
        print('C')
        super().x()


class D(B, C):
    pass
    # def __init__(self):
    # super().__init__()
    #    print('init D')
    #   self.b = {}


class Hello:
    pass


a = 'D'

d = D()
d.x()
print(D.mro())
# from sortedcollections import ValueSortedDict, SortedDict
#
# def ValueS():
#     a=SortedDict()
#     a=ValueSortedDict()
# def ValueA():
#     a= {}
#
# import timeit
# print(timeit.timeit('ValueS()', number=50000,setup="from __main__ import ValueS"))
# print(timeit.timeit('ValueA()', number=103000,setup="from __main__ import ValueA"))
