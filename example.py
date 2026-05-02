from derivator import *


def test_matmul() -> None:
    y, A, x = Vector("y"), Matrix("A"), Vector("x")
    i, j, k = Index(3, "i"), Index(3, "j"), Index(3, "k")  # can we reuse j?

    y[i] = A[i, j] * x[j]
    print(y.diff(x[k]))


test_matmul()
