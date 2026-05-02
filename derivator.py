from typing import Union, Tuple, Any
from abc import ABC, abstractmethod

_used_names: dict[int, str] = dict()


def _add_name(obj: Any, name: str) -> None:
    assert (
        name not in _used_names.values()
    ), "Name already used"  # Might get slow when we have too many names?
    _used_names[id(obj)] = name


def _remove_name(obj: Any) -> None:
    del _used_names[id(obj)]


class Index:
    def __init__(self, bound: int, name: str) -> None:
        self.bound = bound
        _add_name(self, name)

    def __lt__(self, other: "Index") -> bool:
        return _used_names[id(self)] < _used_names[id(other)]

    def __del__(self) -> None:
        _remove_name(self)

    def __str__(self) -> str:
        return _used_names[id(self)]


def _tuplify_index(indices: Union[Index, Tuple[Index, ...]]) -> Tuple[Index, ...]:
    if isinstance(indices, Index):
        indices = (indices,)
    return indices


def _get_index_counts(indices: Tuple[Index, ...]) -> dict[Index, int]:
    index_counts: dict[Index, int] = {}
    for i in indices:
        index_counts.setdefault(i, 0)
        index_counts[i] += 1
    return index_counts


def _get_free_indices(indices: Tuple[Index, ...]) -> set[Index]:
    index_counts = _get_index_counts(indices)
    return {key for key, val in index_counts.items() if val == 1}


def _get_dummy_indices(indices: Tuple[Index, ...]) -> set[Index]:
    index_counts = _get_index_counts(indices)
    return {key for key, val in index_counts.items() if val != 1}


def _check_indices(ours: Tuple[Index, ...], target: Tuple[Index, ...]) -> None:
    if set(ours) & set(target):
        raise ValueError("Re-use of indices is not supported")


class IndexedExpr(ABC):
    @abstractmethod
    def get_free_indices(self) -> set[Index]:
        pass

    @abstractmethod
    def diff(self, target: TensorIndexing) -> IndexedExpr:
        pass

    def __mul__(self, other: IndexedExpr) -> Product:
        return Product(self, other)

    def __add__(self, other: IndexedExpr) -> Sum:
        return Sum(self, other)

    @abstractmethod
    def __str__(self) -> str:
        pass


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

    def diff(self, target: TensorIndexing) -> IndexedExpr:
        t1 = self.lhs * self.rhs.diff(target)
        t2 = self.lhs.diff(target) * self.rhs
        return t1 + t2

    def __str__(self) -> str:
        return f"{self.lhs} * {self.rhs}"


class Sum(IndexedExpr):
    def __init__(self, lhs: IndexedExpr, rhs: IndexedExpr) -> None:
        lhs_free = lhs.get_free_indices()
        rhs_free = rhs.get_free_indices()
        assert (
            lhs_free == rhs_free
        ), f"LHS({lhs_free}) and RHS({rhs_free}) free indices don't match"
        self.lhs = lhs
        self.rhs = rhs

    def get_free_indices(self) -> set[Index]:
        return self.lhs.get_free_indices()

    def diff(self, target: TensorIndexing) -> IndexedExpr:
        return self.lhs.diff(target) + self.rhs.diff(target)

    def __str__(self) -> str:
        return f"({self.lhs} + {self.rhs})"


class Delta(IndexedExpr):
    def __init__(self, i1: Index, i2: Index) -> None:
        self.i1 = i1
        self.i2 = i2

    def get_free_indices(self) -> set[Index]:
        if self.i1 == self.i2:
            return set()
        else:
            return {self.i1, self.i2}

    def diff(self, target: TensorIndexing) -> IndexedExpr:
        indices_tuple = (self.i1, self.i2)
        _check_indices(indices_tuple, target.indices)
        new_indices = indices_tuple + target.indices
        return _Zero(len(new_indices)).__getitem__(new_indices)

    def __str__(self) -> str:
        return f"delta({self.i1}, {self.i2})"


class TensorIndexing(IndexedExpr):
    def __init__(self, tensor: "_Tensor", indices: Tuple[Index, ...]) -> None:
        assert (
            len(indices) == tensor.rank
        ), f"Index count({len(indices)}) doesn't match rank({tensor.rank})"
        self.tensor = tensor
        self.indices = indices

    def get_free_indices(self) -> set[Index]:
        return _get_free_indices(self.indices)

    def diff(self, target: TensorIndexing) -> IndexedExpr:
        _check_indices(self.indices, target.indices)
        if target.tensor == self.tensor:
            if self.tensor.rank == 0:
                return _One(0).__getitem__(())
            expr: IndexedExpr = Delta(
                self.indices[0], target.indices[0]
            )
            for i, j in zip(self.indices[1:], target.indices[1:]):
                expr = expr * Delta(i, j)
            return expr
        else:
            new_indices = self.indices + target.indices
            return _Zero(len(new_indices)).__getitem__(new_indices)

    def __str__(self) -> str:
        return f"{self.tensor}[{",".join([str(i) for i in self.indices])}]"


class Equality:
    def __init__(self, indices: Tuple[Index, ...], rhs: IndexedExpr) -> None:
        dummy = _get_dummy_indices(indices)
        assert not dummy, f"Dummy indices detected in assignment LHS: {dummy}"
        rhs_free_indices = rhs.get_free_indices()
        assert rhs_free_indices == set(
            indices
        ), f"RHS indices({rhs_free_indices}) don't match LHS({indices})"
        self.rhs = rhs
        self.free_indices = indices  # we don't really need this i think


class _Tensor(ABC):
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def __getitem__(self, indices: Union[Index, Tuple[Index, ...]]) -> TensorIndexing:
        indices = _tuplify_index(indices)
        return TensorIndexing(self, indices)

    def __setitem__(
        self, indices: Union[Index, Tuple[Index, ...]], expr: IndexedExpr
    ) -> None:
        indices = _tuplify_index(indices)
        self.value = Equality(indices, expr)

    def diff(self, x: TensorIndexing) -> IndexedExpr:
        if not self.value:
            raise ValueError("Tensor has not been assigned an expr")

        return self.value.rhs.diff(x)

    @abstractmethod
    def __str__(self) -> str:
        pass


class Tensor(_Tensor):
    def __init__(self, rank: int, name: str) -> None:
        super().__init__(rank)
        _add_name(self, name)

    def __del__(self) -> None:
        _remove_name(self)

    def __str__(self) -> str:
        return _used_names[id(self)]


class Scalar(Tensor):
    def __init__(self, name: str) -> None:
        super().__init__(0, name)


class Vector(Tensor):
    def __init__(self, name: str) -> None:
        super().__init__(1, name)


class Matrix(Tensor):
    def __init__(self, name: str) -> None:
        super().__init__(2, name)


class _Zero(_Tensor):
    def __init__(self, rank: int) -> None:
        super().__init__(rank)

    def __str__(self) -> str:
        return "0"

class _One(_Tensor):
    def __init__(self, rank: int) -> None:
        super().__init__(rank)

    def __str__(self) -> str:
        return "1"
