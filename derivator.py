from typing import Union, Tuple, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import functools
import operator

_used_names: dict[int, str] = dict()


class Context:
    remove_zeros_and_ones: bool = True
    simplify_deltas: bool = False


def _add_name(obj: Any, name: str) -> None:
    assert (
        name not in _used_names.values()
    ), "Name already used"  # Might get slow when we have too many names?
    _used_names[id(obj)] = name


def _remove_name(obj: Any) -> None:
    del _used_names[id(obj)]


class Index:
    def __init__(self, name: str) -> None:
        _add_name(self, name)

    def __lt__(self, other: "Index") -> bool:
        return _used_names[id(self)] < _used_names[id(other)]

    def __del__(self) -> None:
        _remove_name(self)

    def __str__(self) -> str:
        return _used_names[id(self)]

    def __repr__(self) -> str:
        return self.__str__()


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


class _IndexedExpr(ABC):
    @abstractmethod
    def get_free_indices(self) -> set[Index]:
        pass

    @abstractmethod
    def diff(self, target: _TensorIndexing) -> _IndexedExpr:
        pass

    def __mul__(self, other: _IndexedExpr) -> _IndexedExpr:
        if Context.remove_zeros_and_ones:
            if (
                isinstance(self, _TensorIndexing) and isinstance(self.tensor, _Zero)
            ) or (
                isinstance(other, _TensorIndexing) and isinstance(other.tensor, _Zero)
            ):
                self_free = self.get_free_indices()
                other_free = other.get_free_indices()
                return _make_indexed(_Zero, tuple(self_free ^ other_free))
            if isinstance(self, _TensorIndexing) and isinstance(self.tensor, _One):
                return other
            if isinstance(other, _TensorIndexing) and isinstance(other.tensor, _One):
                return self
        if isinstance(self, _Product) and isinstance(other, _Product):
            return _Product(self.operands + other.operands)
        if isinstance(self, _Product):
            return _Product(self.operands + [other])
        if isinstance(other, _Product):
            return _Product([self] + other.operands)
        return _Product([self, other])

    def __add__(self, other: _IndexedExpr) -> _IndexedExpr:
        self_free = self.get_free_indices()
        other_free = other.get_free_indices()
        assert (
            self_free == other_free
        ), f"LHS({self_free}) and RHS({other_free}) free indices don't match"

        if Context.remove_zeros_and_ones:
            self_zero = isinstance(self, _TensorIndexing) and isinstance(
                self.tensor, _Zero
            )
            other_zero = isinstance(other, _TensorIndexing) and isinstance(
                other.tensor, _Zero
            )
            if self_zero and other_zero:
                return self  # return any
            if self_zero:
                return other
            if other_zero:
                return self
        if isinstance(self, _Sum) and isinstance(other, _Sum):
            return _Sum(self.operands + other.operands)
        if isinstance(self, _Sum):
            return _Sum(self.operands + [other])
        if isinstance(other, _Sum):
            return _Sum([self] + other.operands)
        return _Sum([self, other])

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class _Product(_IndexedExpr):
    # this will always have size >= 2
    operands: list[_IndexedExpr]

    def get_free_indices(self) -> set[Index]:
        all_free_indices = []
        for o in self.operands:
            all_free_indices += list(o.get_free_indices())
        free_indices = _get_free_indices(tuple(all_free_indices))
        return free_indices

    def diff(self, target: _TensorIndexing) -> _IndexedExpr:
        # I am 99% sure that target will never have any index repeated from the free indices
        new_indices = tuple(self.get_free_indices() | target.get_free_indices())
        return sum(
            (
                functools.reduce(
                    operator.mul,
                    self.operands[:i] + [op.diff(target)] + self.operands[i + 1 :],
                    _One()[()],
                )
                for i, op in enumerate(self.operands)
            ),
            _make_indexed(_Zero, new_indices),
        )

    def __str__(self) -> str:
        return " * ".join([str(op) for op in self.operands])


@dataclass
class _Sum(_IndexedExpr):
    # this will always have size >= 2
    operands: list[_IndexedExpr]

    def get_free_indices(self) -> set[Index]:
        return self.operands[0].get_free_indices()

    def diff(self, target: _TensorIndexing) -> _IndexedExpr:
        new_indices = tuple(self.get_free_indices() | target.get_free_indices())
        return sum(
            (op.diff(target) for op in self.operands), _make_indexed(_Zero, new_indices)
        )

    def __str__(self) -> str:
        return " + ".join([str(op) for op in self.operands])


@dataclass
class _Delta(_IndexedExpr):
    i1: Index
    i2: Index

    def get_free_indices(self) -> set[Index]:
        if self.i1 == self.i2:
            return set()
        else:
            return {self.i1, self.i2}

    def diff(self, target: _TensorIndexing) -> _IndexedExpr:
        indices_tuple = (self.i1, self.i2)
        _check_indices(indices_tuple, target.indices)
        new_indices = indices_tuple + target.indices
        return _make_indexed(_Zero, new_indices)

    def __str__(self) -> str:
        return f"delta({self.i1}, {self.i2})"


@dataclass
class _TensorIndexing(_IndexedExpr):
    tensor: _Tensor
    indices: Tuple[Index, ...]

    def get_free_indices(self) -> set[Index]:
        return _get_free_indices(self.indices)

    def diff(self, target: _TensorIndexing) -> _IndexedExpr:
        _check_indices(self.indices, target.indices)
        if target.tensor == self.tensor:
            if self.tensor.rank == 0:
                return _One()[()]
            expr: _IndexedExpr = _Delta(self.indices[0], target.indices[0])
            for i, j in zip(self.indices[1:], target.indices[1:]):
                expr = expr * _Delta(i, j)
            return expr
        else:
            new_indices = self.indices + target.indices
            return _make_indexed(_Zero, new_indices)

    def __str__(self) -> str:
        return f"{self.tensor}[{",".join([str(i) for i in self.indices])}]"


class _Equality:
    def __init__(self, indices: Tuple[Index, ...], rhs: _IndexedExpr) -> None:
        dummy = _get_dummy_indices(indices)
        assert not dummy, f"Dummy indices detected in assignment LHS: {dummy}"
        rhs_free_indices = rhs.get_free_indices()
        assert rhs_free_indices == set(
            indices
        ), f"RHS indices({rhs_free_indices}) don't match LHS({indices})"
        self.rhs = rhs
        self.free_indices = indices  # we don't really need this i think


def _make_indexed(
    T: Callable[[int], _Tensor], indices: Tuple[Index, ...]
) -> _IndexedExpr:
    return T(len(indices)).__getitem__(indices)


class _Tensor(ABC):
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def __getitem__(self, indices: Union[Index, Tuple[Index, ...]]) -> _TensorIndexing:
        indices = _tuplify_index(indices)
        assert (
            len(indices) == self.rank
        ), f"Index count({len(indices)}) doesn't match rank({self.rank})"
        return _TensorIndexing(self, indices)

    def __setitem__(
        self, indices: Union[Index, Tuple[Index, ...]], expr: _IndexedExpr
    ) -> None:
        indices = _tuplify_index(indices)
        self.value = _Equality(indices, expr)

    def diff(self, x: _TensorIndexing) -> _IndexedExpr:
        if not self.value:
            raise ValueError("Tensor has not been assigned an expr")

        return self.value.rhs.diff(x)

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.__str__()


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
    def __init__(self) -> None:
        super().__init__(0)

    def __str__(self) -> str:
        return "1"
