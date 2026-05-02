from derivator import Tensor, Index

y, A, x = Tensor(1, "y"), Tensor(2, "A"), Tensor(1, "x")
i, j = Index(3, "i"), Index(3, "j")

y[i] = A[i, j] * x[j]
