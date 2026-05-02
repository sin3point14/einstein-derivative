from typing import Union, Tuple
from abc import ABC, abstractmethod


class Scalar:
    def __init__(self) -> None:
        pass


__used_names: set[str] = set()


class Index:
    def __init__(self, bound: int, name: str) -> None:
        self.bound = bound
        assert name not in __used_names, "Name already used"
        self.name = name
        __used_names.add(name)

    def __lt__(self, other: "Index") -> bool:
        return self.name < other.name


def __tuplify_index(indices: Union[Index, Tuple[Index, ...]]) -> Tuple[Index, ...]:
    if isinstance(indices, Index):
        indices = (indices,)
    return indices


def __get_index_counts(indices: Tuple[Index, ...]) -> dict[Index, int]:
    index_counts: dict[Index, int] = {}
    for i in indices:
        index_counts.setdefault(i, 0)
        index_counts[i] += 1
    return index_counts


def __get_free_indices(indices: Tuple[Index, ...]) -> set[Index]:
    index_counts = __get_index_counts(indices)
    return {key for key, val in index_counts.items() if val == 1}


def __get_dummy_indices(indices: Tuple[Index, ...]) -> set[Index]:
    index_counts = __get_index_counts(indices)
    return {key for key, val in index_counts.items() if val != 1}


class IndexedExpr(ABC):
    @abstractmethod
    def get_free_indices(self) -> set[Index]:
        pass

    def __mul__(self, other: IndexedExpr) -> Product:
        return Product(self, other)

    def __add__(self, other: IndexedExpr) -> Sum:
        return Sum(self, other)


class Product(IndexedExpr):
    def __init__(self, lhs: IndexedExpr, rhs: IndexedExpr) -> None:
        self.lhs = lhs
        self.rhs = rhs

    def get_free_indices(self) -> set[Index]:
        lhs_indices, rhs_indices = (
            self.lhs.get_free_indices(),
            self.rhs.get_free_indices(),
        )
        return lhs_indices ^ rhs_indices


class Sum(IndexedExpr):
    def __init__(self, lhs: IndexedExpr, rhs: IndexedExpr) -> None:
        assert sorted(lhs.get_free_indices()) == sorted(rhs.get_free_indices())
        self.lhs = lhs
        self.rhs = rhs

    def get_free_indices(self) -> set[Index]:
        return self.lhs.get_free_indices()


class TensorIndexing(IndexedExpr):
    def __init__(self, tensor: "Tensor", indices: Tuple[Index, ...]) -> None:
        self.tensor = tensor
        self.indices = indices

    def get_free_indices(self) -> set[Index]:
        return __get_free_indices(self.indices)


class Tensor:
    def __init__(self, rank: int, name: str) -> None:
        self.rank = rank
        assert name not in __used_names, "Name already used"
        self.name = name
        __used_names.add(name)

    def __getitem__(self, indices: Union[Index, Tuple[Index, ...]]) -> TensorIndexing:
        indices = __tuplify_index(indices)
        assert (
            len(indices) == self.rank
        ), f"Index count({len(indices)}) doesn't match rank({self.rank})"
        return TensorIndexing(self, indices)

    def __setitem__(
        self, indices: Union[Index, Tuple[Index, ...]], expr: IndexedExpr
    ) -> None:
        indices = __tuplify_index(indices)
        dummy = __get_dummy_indices(indices)
        assert not dummy, f"Dummy indices detected: {dummy}"

        rhs_free_indices = expr.get_free_indices()
        assert rhs_free_indices == set(
            indices
        ), f"RHS indices({rhs_free_indices}) don't match LHS({indices})"
