from derivator import *


def test_matmul() -> None:
    y, A, x = Vector("y"), Matrix("A"), Vector("x")
    i, j, k = Index("i"), Index("j"), Index("k")  # can we reuse j?

    y[i] = A[i, j] * x[j]
    print(y.diff(x[k]))


def test_scalar() -> None:
    y, a, x = Scalar("y"), Scalar("a"), Scalar("x")
    y[()] = a[()] * x[()]

    print(y.diff(x[()]))


test_matmul()
# test_scalar()
