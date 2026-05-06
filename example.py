from derivator import *
import sys

# sys.setrecursionlimit(20)


def print_diff(statements: list[Tensor]) -> None:
    for s in statements:
        assert s.value
        print(f"{s[s.value.free_indices]} = {s.value.rhs}")


def test_matmul() -> None:
    print("Test matmul")
    A, x, y = Matrix("A"), Vector("x"), Vector("y")
    i, j, k = Index("i"), Index("j"), Index("k")  # can we reuse j?

    y[i] = A[i, j] * x[j]
    print_diff(y.diff(x[k]))
    print()


def test_scalar() -> None:
    print("Test scalar")
    a, x, y = Scalar("a"), Scalar("x"), Scalar("y")
    y[()] = a * x

    print_diff(y.diff(x))
    print()


def test_dot_prod() -> None:
    print("Test dot prod")
    a, x = Vector("a"), Vector("x")
    y = Scalar("y")
    i, j = Index("i"), Index("j")
    y[()] = a[i] * x[i]

    print_diff(y.diff(x[j]))
    print()


def test_stvk_green_strain() -> None:
    mu, lambda_, psi = Scalar("mu"), Scalar("lambda"), Scalar("psi")
    X, E, B, F = Matrix("X"), Matrix("E"), Matrix("B"), Matrix("F")
    a, b, c = Index("a"), Index("b"), Index("c")
    F[a, c] = X[b, a] * B[b, c]  # F = X.T @ B

    e = Index("e")
    E[a, c] = 0.5 * (F[e, a] * F[e, c] - Delta(a, c))

    i, j, k, l = Index("i"), Index("j"), Index("k"), Index("l")
    psi[()] = mu * E[i, j] * E[i, j] + 0.5 * lambda_ * E[i, i] * E[j, j]

    print_diff(psi.diff(F[k, l]))
    # print_diff(diff(psi, X[k, l]))


def test_sign() -> None:
    a, b, x = Scalar("a"), Scalar("b"), Scalar("x")
    x[()] = b * -(a)
    print("Test sign")
    print(x)
    print_diff(x.diff(a))
    x[()] = b - -a
    print(x)
    print_diff(x.diff(a))
    x[()] = b * -2
    print(x)
    print_diff(x.diff(b))
    x[()] = -a - b
    print(x)
    print_diff(x.diff(a))
    x[()] = -(-a - b)
    print(x)
    print_diff(x.diff(b))
    print()


# test_matmul()
# test_scalar()
# test_dot_prod()
test_stvk_green_strain()
# test_sign()
