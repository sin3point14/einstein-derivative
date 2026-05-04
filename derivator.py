from __future__ import annotations
from typing import Union, Tuple, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import functools
import operator
import utils

_used_names: dict[int, str] = dict()


class Context:
    remove_zeros_and_ones: bool = True
    simplify_deltas: bool = True


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


def _listify_index(indices: Union[Index, Tuple[Index, ...]]) -> list[Index]:
    if isinstance(indices, Index):
        indices = (indices,)
    return list(indices)


def _get_index_counts(indices: list[Index]) -> dict[Index, int]:
    index_counts: dict[Index, int] = {}
    for i in indices:
        index_counts.setdefault(i, 0)
        index_counts[i] += 1
    return index_counts


def _get_free_indices(indices: list[Index]) -> set[Index]:
    index_counts = _get_index_counts(indices)
    return {key for key, val in index_counts.items() if val == 1}


def _get_dummy_indices(indices: list[Index]) -> set[Index]:
    index_counts = _get_index_counts(indices)
    return {key for key, val in index_counts.items() if val != 1}


def _check_indices_diff(ours: list[Index], target: list[Index]) -> None:
    common_indices = set(ours) & set(target)
    if common_indices:
        raise ValueError(
            f"Re-use of indices({common_indices}) in differentiation is not supported"
        )


class _IndexedExpr(ABC):
    @abstractmethod
    def get_free_indices(self) -> set[Index]:
        pass

    @abstractmethod
    def get_dummy_indices(self) -> set[Index]:
        pass

    @abstractmethod
    def diff(self, target: "_TensorIndexing") -> "_IndexedExpr":
        pass

    def __mul__(self, other: "_IndexedExpr" | int | float) -> "_IndexedExpr":
        if isinstance(other, int) or isinstance(other, float):
            other = _ImplicitScalar(other).__getitem__(())
        return _Product.create(self, other)

    def __rmul__(self, other: int | float) -> "_IndexedExpr":
        return _Product.create(_ImplicitScalar(other).__getitem__(()), self)

    def __add__(self, other: "_IndexedExpr") -> "_IndexedExpr":
        return _Sum.create(self, other)

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def replace_indices(self, replacements: dict[Index, Index]) -> None:
        pass

    @abstractmethod
    def simplify_deltas(self) -> None:
        pass


class _Product(_IndexedExpr, metaclass=utils.NoPublicConstructor):
    def __init__(self, operands: list[_IndexedExpr]):
        self.operands = operands

    @classmethod
    def create(cls, lhs: _IndexedExpr, rhs: _IndexedExpr) -> _IndexedExpr:
        free_lhs, dummy_lhs = lhs.get_free_indices(), lhs.get_dummy_indices()
        free_rhs, dummy_rhs = rhs.get_free_indices(), rhs.get_dummy_indices()
        common_dummy = dummy_lhs & dummy_rhs
        assert not common_dummy, f"Indices({common_dummy}) appear more than 2 times"
        all_dummy = dummy_rhs | dummy_lhs
        lhs_repeated = free_lhs & all_dummy
        rhs_repeated = free_rhs & all_dummy
        repeated = lhs_repeated | rhs_repeated
        assert not repeated, f"Indices({repeated}) appear more than 2 times"

        if Context.remove_zeros_and_ones:
            if (isinstance(lhs, _TensorIndexing) and isinstance(lhs.tensor, _Zero)) or (
                isinstance(rhs, _TensorIndexing) and isinstance(rhs.tensor, _Zero)
            ):
                self_free = lhs.get_free_indices()
                other_free = rhs.get_free_indices()
                return _make_indexed(_Zero, list(self_free ^ other_free))
            if isinstance(lhs, _TensorIndexing) and isinstance(lhs.tensor, _One):
                return rhs
            if isinstance(rhs, _TensorIndexing) and isinstance(rhs.tensor, _One):
                return lhs

        if isinstance(lhs, _Product) and isinstance(rhs, _Product):
            return _Product._create(lhs.operands + rhs.operands)
        if isinstance(lhs, _Product):
            return _Product._create(lhs.operands + [rhs])
        if isinstance(rhs, _Product):
            return _Product._create([lhs] + rhs.operands)
        return _Product._create([lhs, rhs])

    def get_free_indices(self) -> set[Index]:
        all_free_indices = []
        for o in self.operands:
            all_free_indices += list(o.get_free_indices())
        free_indices = _get_free_indices(all_free_indices)
        return free_indices

    def get_dummy_indices(self) -> set[Index]:
        old_dummy_indices = set()
        all_free_indices = []
        for o in self.operands:
            all_free_indices += list(o.get_free_indices())
            old_dummy_indices |= o.get_dummy_indices()
        new_dummy_indices = _get_dummy_indices(all_free_indices)
        return new_dummy_indices | old_dummy_indices

    def diff(self, target: _TensorIndexing) -> _IndexedExpr:
        # If target will have any index repeated from the free indices, it will be caught in one of the .diff calls
        new_indices = list(self.get_free_indices() | target.get_free_indices())
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

    def replace_indices(self, replacements: dict[Index, Index]) -> None:
        for o in self.operands:
            o.replace_indices(replacements)

    def simplify_deltas(self) -> None:
        dummy_indices = self.get_dummy_indices()
        deletion_set = set()
        replacements: dict[Index, Index] = {}
        reverse_replacements: dict[Index, Index] = {}

        def add_replacement(old: Index, new: Index) -> None:
            # already have j -> k (replacements), k -> j (reverse_replacements)
            # adding i(old) -> j(new)
            # want i -> k (replacements), k -> i (reverse_replacements)
            if new in replacements:
                new_replacement = replacements[new]
                del replacements[new]
                del reverse_replacements[new_replacement]
                new = new_replacement
            # already have i -> j (replacements), j -> i (reverse_replacements)
            # adding j(old) -> k(new)
            # want i -> k (replacements), k -> i (reverse_replacements)
            if old in reverse_replacements:
                old_reverse_replacement = reverse_replacements[old]
                del reverse_replacements[old]
                del replacements[old_reverse_replacement]
                old = old_reverse_replacement
            replacements[old] = new
            reverse_replacements[new] = old

        for n, o in enumerate(self.operands):
            if isinstance(o, Delta):
                deletion = False
                if o.i1 in dummy_indices:
                    add_replacement(o.i1, o.i2)
                    deletion = True
                if not deletion and o.i2 in dummy_indices:
                    add_replacement(o.i2, o.i1)
                    deletion = True
                if deletion:
                    deletion_set.add(n)

        self.operands = [
            i for j, i in enumerate(self.operands) if j not in deletion_set
        ]

        for o in self.operands:
            o.replace_indices(replacements)


class _Sum(_IndexedExpr, metaclass=utils.NoPublicConstructor):
    def __init__(self, operands: list[_IndexedExpr]):
        self.operands = operands

    @classmethod
    def create(cls, lhs: _IndexedExpr, rhs: _IndexedExpr) -> _IndexedExpr:
        lhs_free = lhs.get_free_indices()
        rhs_free = rhs.get_free_indices()
        assert (
            lhs_free == rhs_free
        ), f"LHS({lhs_free}) and RHS({rhs_free}) free indices don't match"

        if Context.remove_zeros_and_ones:
            self_zero = isinstance(lhs, _TensorIndexing) and isinstance(
                lhs.tensor, _Zero
            )
            other_zero = isinstance(rhs, _TensorIndexing) and isinstance(
                rhs.tensor, _Zero
            )
            if self_zero and other_zero:
                return lhs  # return any
            if self_zero:
                return rhs
            if other_zero:
                return lhs
        if isinstance(lhs, _Sum) and isinstance(rhs, _Sum):
            return _Sum._create(lhs.operands + rhs.operands)
        if isinstance(lhs, _Sum):
            return _Sum._create(lhs.operands + [rhs])
        if isinstance(rhs, _Sum):
            return _Sum._create([lhs] + rhs.operands)
        return _Sum._create([lhs, rhs])

    def get_free_indices(self) -> set[Index]:
        return self.operands[0].get_free_indices()

    def get_dummy_indices(self) -> set[Index]:
        all_dummy_indices = set()
        for o in self.operands:
            all_dummy_indices |= o.get_dummy_indices()
        return all_dummy_indices

    def diff(self, target: _TensorIndexing) -> _IndexedExpr:
        new_indices = list(self.get_free_indices() | target.get_free_indices())
        return sum(
            (op.diff(target) for op in self.operands), _make_indexed(_Zero, new_indices)
        )

    def __str__(self) -> str:
        return " + ".join([str(op) for op in self.operands])

    def replace_indices(self, replacements: dict[Index, Index]) -> None:
        for o in self.operands:
            o.replace_indices(replacements)

    def simplify_deltas(self) -> None:
        for o in self.operands:
            o.simplify_deltas()


class Delta(_IndexedExpr):
    i1: Index
    i2: Index

    def __init__(self, i1: Index, i2: Index):
        assert i1 != i2, f"Delta index({i1}) cannot be repeated"
        self.i1 = i1
        self.i2 = i2

    def get_free_indices(self) -> set[Index]:
        if self.i1 == self.i2:
            return set()
        else:
            return {self.i1, self.i2}

    def get_dummy_indices(self) -> set[Index]:
        if self.i1 == self.i2:
            return {self.i1}
        else:
            return set()

    def diff(self, target: _TensorIndexing) -> _IndexedExpr:
        indices_list = [self.i1, self.i2]
        _check_indices_diff(indices_list, target.indices)
        new_indices = indices_list + target.indices
        return _make_indexed(_Zero, new_indices)

    def __str__(self) -> str:
        return f"delta({self.i1}, {self.i2})"

    def get_children(self) -> list[_IndexedExpr]:
        return []

    def replace_indices(self, replacements: dict[Index, Index]) -> None:
        for old, new in replacements.items():
            # it is guaranteed by construction that only one of these ifs will be hit
            if self.i1 == old:
                self.i1 = new
            if self.i2 == old:
                self.i2 = new

    def simplify_deltas(self) -> None:
        return


@dataclass
class _TensorIndexing(_IndexedExpr):
    tensor: _Tensor
    indices: list[Index]

    def __init__(self, tensor: _Tensor, indices: list[Index]) -> None:
        assert (
            len(indices) == tensor.rank
        ), f"Index count({len(indices)}) doesn't match rank({tensor.rank})"
        counts = _get_index_counts(indices)
        for i, c in counts.items():
            if c > 2:
                raise ValueError(f"Index {i} appears more the 2 times")
        self.tensor = tensor
        self.indices = indices

    def get_free_indices(self) -> set[Index]:
        return _get_free_indices(self.indices)

    def get_dummy_indices(self) -> set[Index]:
        return _get_dummy_indices(self.indices)

    def diff(self, target: _TensorIndexing) -> _IndexedExpr:
        _check_indices_diff(self.indices, target.indices)
        if target.tensor == self.tensor:
            if self.tensor.rank == 0:
                return _One()[()]
            expr: _IndexedExpr = Delta(self.indices[0], target.indices[0])
            for i, j in zip(self.indices[1:], target.indices[1:]):
                expr = expr * Delta(i, j)
            return expr
        else:
            new_indices = self.indices + target.indices
            return _make_indexed(_Zero, new_indices)

    def __str__(self) -> str:
        return f"{self.tensor}[{",".join([str(i) for i in self.indices])}]"

    def get_children(self) -> list[_IndexedExpr]:
        return []

    def replace_indices(self, replacements: dict[Index, Index]) -> None:
        for old, new in replacements.items():
            for n in range(len(self.indices)):
                if self.indices[n] == old:
                    self.indices[n] = new

    def simplify_deltas(self) -> None:
        return


class _Equality:
    def __init__(self, indices: list[Index], rhs: _IndexedExpr) -> None:
        dummy = _get_dummy_indices(indices)
        assert not dummy, f"Dummy indices({dummy}) detected in assignment LHS"
        rhs_free_indices = rhs.get_free_indices()
        assert rhs_free_indices == set(
            indices
        ), f"RHS indices({rhs_free_indices}) don't match LHS({indices})"
        self.rhs = rhs
        self.free_indices = indices  # we don't really need this i think


def _make_indexed(T: Callable[[int], _Tensor], indices: list[Index]) -> _IndexedExpr:
    return T(len(indices)).__getitem__(tuple(indices))


class _Tensor(ABC):
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def __getitem__(self, indices: Union[Index, Tuple[Index, ...]]) -> _TensorIndexing:
        indices_list = _listify_index(indices)
        return _TensorIndexing(self, indices_list)

    def __setitem__(
        self, indices: Union[Index, Tuple[Index, ...]], expr: _IndexedExpr
    ) -> None:
        indices_list = _listify_index(indices)
        self.value = _Equality(indices_list, expr)

    def diff(self, x: _TensorIndexing) -> _IndexedExpr:
        if not self.value:
            raise ValueError("Tensor has not been assigned an expr")
        diff_expr = self.value.rhs.diff(x)
        if Context.simplify_deltas:
            diff_expr.simplify_deltas()
        return diff_expr

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

# TODO: Convert to _One if 1 is passed
class _ImplicitScalar(_Tensor):
    def __init__(self, num: int | float) -> None:
        super().__init__(0)
        self.num = num

    def __str__(self) -> str:
        return str(self.num)


class _One(_ImplicitScalar):
    def __init__(self) -> None:
        super().__init__(1)
