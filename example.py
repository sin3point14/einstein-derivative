from derivator import *


def test_matmul() -> None:
    print("Test matmul")
    A, x = Matrix("A"), Vector("x")
    i, j, k = Index("i"), Index("j"), Index("k")  # can we reuse j?

    y = A[i, j] * x[j]
    print(diff(y, x[k]))
    print()


def test_scalar() -> None:
    print("Test scalar")
    a, x = Scalar("a"), Scalar("x")
    y = a * x

    print(diff(y, x))
    print()


def test_dot_prod() -> None:
    print("Test dot prod")
    a, x = Vector("a"), Vector("x")
    i, j = Index("i"), Index("j")
    y = a[i] * x[i]

    print(diff(y, x[j]))
    print()


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
    a, b = Scalar("a"), Scalar("b")
    x = b * -(a)
    print("Test sign")
    print(x)
    print(diff(x, a))
    x = b - -a
    print(x)
    print(diff(x, a))
    x = b * -2
    print(x)
    print(diff(x, b))
    x = -a - b
    print(x)
    print(diff(x, a))
    x = -(-a - b)
    print(x)
    print(diff(x, b))
    print()


test_matmul()
test_scalar()
test_dot_prod()
# test_stvk_green_strain()
test_sign()
