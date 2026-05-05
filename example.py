from derivator import *


def test_matmul() -> None:
    A, x = Matrix("A"), Vector("x")
    i, j, k = Index("i"), Index("j"), Index("k")  # can we reuse j?

    y = A[i, j] * x[j]
    print(diff(y, x[k]))


def test_scalar() -> None:
    a, x = Scalar("a"), Scalar("x")
    y = a * x

    print(diff(y, x))


def test_dot_prod() -> None:
    a, x = Vector("a"), Vector("x")
    i, j = Index("i"), Index("j")
    y = a[i] * x[i]

    print(diff(y, x[j]))


# def test_stvk_green_strain() -> None:
#     mu, lambda_ = Scalar("mu"), Scalar("lambda")
#     X, E, B, F = Matrix("X"), Matrix("E"), Matrix("B"), Matrix("F")
#     a, b, c = Index("a"), Index("b"), Index("c")
#     F[a, c] = X[b, a] * B[b, c]  # F = X.T @ B

#     d = Index("d")
#     E[a, c] = 0.5 * (F[d, a] * F[d, c] - Delta(a, c))

#     i, j, k, l = Index("i"), Index("j"), Index("k"), Index("l")
#     psi = mu * E[i, j] * E[i, j] + 0.5 * lambda_ * E[i, i] * E[j, j]
#     print(diff(psi, X[k]))


def test_sign() -> None:
    a = Scalar("a")
    x = -(a)
    print(x)


test_matmul()
test_scalar()
test_dot_prod()
# test_stvk_green_strain()
test_sign()
