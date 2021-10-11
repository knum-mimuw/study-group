# important decorators and magic methods
import math
import functools
import copy
import operator


@functools.total_ordering
class Fraction:
    def __init__(self, a, b):
        if b == 0:
            raise ValueError
        self.a = a
        self.b = b
        self.reduce()

    def reduce(self):

        if self.a == 0:
            self.b = 1
            return self

        if self.a * self.b < 0:
            self.a = -abs(self.a)
            self.b = abs(self.b)
        else:
            self.a = abs(self.a)
            self.b = abs(self.b)

        if math.gcd(self.a, self.b) != 1:
            self.a //= math.gcd(self.a, self.b)
            self.b //= math.gcd(self.a, self.b)

        return self

    @functools.singledispatchmethod
    def __add__(self, other):
        return Fraction(self.a * other.b + self.b * other.a, other.b * self.b).reduce()

    @__add__.register(int)
    def _(self, other):
        return self + Fraction(other, 1)

    def __neg__(self):
        return Fraction(-self.a, self.b).reduce()

    @functools.singledispatchmethod
    def __mul__(self, other):
        return Fraction(self.a * other.a, self.b * other.b).reduce()

    # without @singledispatchmethod
    # def __mul__(self, other):
    #     if isinstance(other, int):
    #         other = Fraction(other, 1)
    #     return Fraction(self.a * other.a, self.b * other.b).reduce()

    @__mul__.register(int)
    def _(self, other):
        return Fraction(self.a * other, self.b).reduce()

    def __invert__(self):
        return Fraction(self.b, self.a).reduce()

    def __str__(self):
        return f'{self.a}/{self.b}'

    def __abs__(self):
        return Fraction(abs(self.a), abs(self.b)).reduce()

    @functools.singledispatchmethod
    def __eq__(self, other):
        self.reduce()
        other.reduce()
        return self.a == other.a and self.b == other.b

    @__eq__.register(int)
    def _(self, other):
        self.reduce()
        return self.a == other and self.b == 1

    @functools.singledispatchmethod
    def __gt__(self, other):
        return not self == other and self.a / self.b > other.a / other.b

    @__gt__.register(int)
    def _(self, other):
        return not self == other and self.a / self.b > other

    @functools.singledispatchmethod
    def __truediv__(self, other):
        return self * ~other

    @__truediv__.register(int)
    def _(self, other):
        return Fraction(self.a, self.b * other)

    def __sub__(self, other):
        return self + -other

    def to_common_divisor(self, other):
        gcd_ = math.gcd(self.b, other.b)
        return Fraction(self.a*other.b/gcd_, self.b*other.b/gcd_), Fraction(other.a*self.b/gcd_, self.b*other.b/gcd_)


class Matrix:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __mul__(self, other):
        return Matrix(list(map(lambda row: [a * other for a in row], self.values)))

    @property
    def shape(self):
        if not self.values:
            return 0, 0
        return len(self.values), len(self.values[0])

    @shape.setter
    def shape(self, value):
        print('you cannot set shape value.')
        raise AttributeError('dupa')

    def __getitem__(self, item):
        return self.values[item[0]][item[1]]

    def del_row(self, i):
        del self.values[i]

    def del_col(self, i):
        for row in self.values:
            del row[i]

    def transpose(self):
        return Matrix(list(zip(*self.values)))

    def adjugate(self):
        return Matrix(
            [[self.minor(j, i) * (-1) ** (i + j) for i in range(len(self))] for j in range(len(self))]).transpose()

    def is_squared(self):
        return len(self) == len(self.values[0])

    def minor(self, i, j):
        if not self.is_squared():
            raise ValueError('Matrix is not squared.')
        m = Matrix(copy.deepcopy(self.values))
        m.del_row(i)
        m.del_col(j)
        if len(m) == 1:
            return m[0, 0]
        return m.determinant()

    def determinant(self):
        if not self.is_squared():
            raise ValueError('Matrix is not squared.')
        if len(self) == 1:
            return self[0, 0]
        if len(self) == 2:
            return self[0, 0]*self[1, 1] - self[0, 1]*self[1, 0]
        return functools.reduce(operator.add, [a * self.minor(0, j) * (-1) ** j for j, a in enumerate(self.values[0])])

    def __invert__(self):
        if not self.is_squared():
            raise ValueError('Matrix is not squared.')
        if len(self) == 1:
            return self.__class__([[self[0, 0].__invert__()]])
        return self.adjugate() * (~self.determinant())

    def __str__(self):
        return '\n'.join(['| ' + ' '.join([str(el) for el in row]) + ' |' for row in self.values])

    def flip_rows(self, i, j):
        self.values[i], self.values[j] = self.values[j], self.values[i]

    def flip_columns(self, i, j):
        for row in self.values:
            row[i], row[j] = row[j], row[i]

    def __sub__(self, other):
        values = [[a-b for a, b in zip(row1, row2)] for row1, row2 in zip(self.values, other.values)]
        return Matrix(values)

    def get_row(self, i):
        return self.values[i]

    def get_column(self, i):
        return [row[i] for row in self.values]

    def __matmul__(self, other):
        values = [[functools.reduce(operator.add, [a*b for a, b in zip(self.get_row(i), other.get_column(j))])
                   for i in range(len(self))] for j in range(len(other.values[0]))]
        return Matrix(values)

    @staticmethod
    def wtf():
        print('wtf')

    @classmethod
    def what_is_that(cls):
        print(f'class: {cls}')


class IdentityMatrix(Matrix):
    def __init__(self, n):
        values = [[Fraction(int(i == j), 1) for i in range(n)] for j in range(n)]
        super().__init__(values)


if __name__ == '__main__':
    id_matrix = IdentityMatrix(4)
    id_matrix.what_is_that()
