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


def test_dot_prod() -> None:
    y, a, x = Scalar("y"), Vector("a"), Vector("x")
    i, j = Index("i"), Index("j")
    y[()] = a[i] * x[i]

    print(y.diff(x[j]))


def test_stvk_green_strain() -> None:
    psi, mu, lambda_ = Scalar("psi"), Scalar("mu"), Scalar("lambda")
    E = Matrix("E")
    i, j, k, l = Index("i"), Index("j"), Index("k"), Index("l")

    psi[()] = mu[()] * E[i, j] * E[i, j] + 0.5 * lambda_[()] * E[i, i] * E[j, j]
    print(psi.diff(E[k, l]))


test_matmul()
test_scalar()
test_dot_prod()
test_stvk_green_strain()
