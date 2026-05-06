from __future__ import annotations
from typing import Union, Tuple, Any, Callable, Optional, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import functools
import operator
import utils
from enum import StrEnum

_used_tensor_names: set[str] = set()
_used_index_names: set[str] = set()


class Context:
    remove_zeros_and_ones: bool = True
    simplify_deltas: bool = True


def _add_name(name: str, cls: Type[object]) -> None:
    assert name, "Empty name"
    if cls == Tensor:
        assert (
            name not in _used_tensor_names
        ), f"Tensor name({name}) already used, don't define Tensor with name starting with 'd'"
        _used_tensor_names.add(name)
    elif cls == Index:
        assert name not in _used_index_names, f"Index name({name}) already used"
        _used_index_names.add(name)


def _remove_name(name: str, cls: Type[object]) -> None:
    # Just in case exception is raised in _add_name
    if cls == Tensor and name in _used_tensor_names:
        _used_tensor_names.remove(name)
    if cls == Index and name in _used_index_names:
        _used_index_names.remove(name)


def _promote_scalars(s: IndexedExpr | Scalar | int | float) -> IndexedExpr:
    if isinstance(s, int) or isinstance(s, float):
        s = _ImplicitScalar(s)
    if isinstance(s, Scalar):
        return s.__getitem__(())
    return s


class Index:
    def __init__(self, name: str) -> None:
        _add_name(name, Index)
        self.name = name

    def __lt__(self, other: "Index") -> bool:
        return self.name < other.name

    def __del__(self) -> None:
        _remove_name(self.name, Index)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()


def _listify_index(
    indices: Union[Index, Tuple[Index, ...], list[Index]],
) -> list[Index]:
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


class _Sign(StrEnum):
    Plus = "+"
    Minus = "-"

    def flip(self) -> _Sign:
        if self == _Sign.Minus:
            return _Sign.Plus
        elif self == _Sign.Plus:
            return _Sign.Minus
        else:
            raise ValueError("Unreachable")

    def to_scalar(self) -> _ImplicitScalar:
        if self == _Sign.Minus:
            return _ImplicitScalar(-1)
        elif self == _Sign.Plus:
            return _ImplicitScalar(1)
        else:
            raise ValueError("Unreachable")


class IndexedExpr(ABC):
    def __init__(self, sign: _Sign = _Sign.Plus):
        self.sign = sign

    def __neg__(self) -> IndexedExpr:
        new_expr = copy.copy(self)
        new_expr.sign = self.sign.flip()
        return new_expr

    @abstractmethod
    def get_free_indices(self) -> set[Index]:
        pass

    @abstractmethod
    def get_dummy_indices(self) -> set[Index]:
        pass

    def _diff(
        self, target: "_TensorIndexing", dependencies: list[Tensor]
    ) -> tuple[list[Tensor], "IndexedExpr"]:
        dependencies, expr = self._signless_diff(target, dependencies)
        return dependencies, self.sign.to_scalar() * expr

    @abstractmethod
    def _signless_diff(
        self, target: "_TensorIndexing", dependencies: list[Tensor]
    ) -> tuple[list[Tensor], "IndexedExpr"]:
        pass

    def __mul__(self, other: "IndexedExpr" | int | float | Scalar) -> "IndexedExpr":
        return _Product.create(self, _promote_scalars(other))

    def __rmul__(self, other: int | float) -> "IndexedExpr":
        return _Product.create(_promote_scalars(other), self)

    def __add__(self, other: "IndexedExpr" | Scalar | int | float) -> "IndexedExpr":
        return _Sum.create(self, _promote_scalars(other))

    def __sub__(self, other: "IndexedExpr" | Scalar | int | float) -> "IndexedExpr":
        other = -_promote_scalars(other)
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

    def add_sign(self, expr: str, wrap_brackets: bool) -> "str":
        if self.sign == _Sign.Plus:
            return expr
        else:
            if wrap_brackets:
                return f"- ({expr})"
            else:
                return f"- {expr}"


class _Product(IndexedExpr, metaclass=utils.NoPublicConstructor):
    def __init__(self, operands: list[IndexedExpr]):
        self.operands = operands
        net_sign = _Sign.Plus
        for o in self.operands:
            if o.sign == _Sign.Minus:
                o.sign = _Sign.Plus
                net_sign = net_sign.flip()
        super().__init__(net_sign)

    @classmethod
    def create(cls, lhs: IndexedExpr, rhs: IndexedExpr) -> IndexedExpr:
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
            if is_zero(lhs) or is_zero(rhs):
                self_free = lhs.get_free_indices()
                other_free = rhs.get_free_indices()
                return _make_indexed(_Zero, list(self_free ^ other_free))
            if is_one(lhs):
                if lhs.sign == _Sign.Minus:
                    rhs = rhs.__neg__()
                return rhs
            if is_one(rhs):
                if rhs.sign == _Sign.Minus:
                    lhs = lhs.__neg__()
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

    def _signless_diff(
        self, target: _TensorIndexing, dependencies: list[Tensor]
    ) -> tuple[list[Tensor], "IndexedExpr"]:
        # If target will have any index repeated from the free indices, it will be caught in one of the .diff calls
        new_indices = list(self.get_free_indices() | target.get_free_indices())
        t = _make_indexed(_Zero, new_indices)

        for i, op in enumerate(self.operands):
            dependencies, diff = op._diff(target, dependencies)
            t += functools.reduce(
                operator.mul,
                self.operands[:i] + [diff] + self.operands[i + 1 :],
                _ImplicitScalar(1),
            )
        return dependencies, t

    def __str__(self) -> str:
        return self.add_sign(" * ".join([str(op) for op in self.operands]), False)

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
                    dummy_indices.remove(o.i1)
                    deletion = True
                if not deletion and o.i2 in dummy_indices:
                    add_replacement(o.i2, o.i1)
                    dummy_indices.remove(o.i2)
                    deletion = True
                if deletion:
                    deletion_set.add(n)

        self.operands = [
            i for j, i in enumerate(self.operands) if j not in deletion_set
        ]

        for o in self.operands:
            o.replace_indices(replacements)

        for o in self.operands:
            o.simplify_deltas()


class _Sum(IndexedExpr, metaclass=utils.NoPublicConstructor):
    def __init__(self, operands: list[IndexedExpr]):
        self.operands = operands
        super().__init__()

    @classmethod
    def create(cls, lhs: IndexedExpr, rhs: IndexedExpr) -> IndexedExpr:
        lhs_free = lhs.get_free_indices()
        rhs_free = rhs.get_free_indices()
        assert (
            lhs_free == rhs_free
        ), f"LHS({lhs_free}) and RHS({rhs_free}) free indices don't match"

        if Context.remove_zeros_and_ones:
            lhs_zero = is_zero(lhs)
            rhs_zero = is_zero(rhs)
            if lhs_zero and rhs_zero:
                return lhs  # return any
            if lhs_zero:
                return rhs
            if rhs_zero:
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

    def _signless_diff(
        self, target: _TensorIndexing, dependencies: list[Tensor]
    ) -> tuple[list[Tensor], IndexedExpr]:
        new_indices = list(self.get_free_indices() | target.get_free_indices())
        t = _make_indexed(_Zero, new_indices)
        for op in self.operands:
            dependencies, diff = op._diff(target, dependencies)
            t += diff
        return dependencies, t

    def __str__(self) -> str:
        ret = str(self.operands[0])
        for op in self.operands[1:]:
            if op.sign == _Sign.Plus:
                ret += f" + {op}"
            else:
                ret += f" {op}"
        ret = f"({ret})"
        return self.add_sign(ret, True)

    def replace_indices(self, replacements: dict[Index, Index]) -> None:
        for o in self.operands:
            o.replace_indices(replacements)

    def simplify_deltas(self) -> None:
        for o in self.operands:
            o.simplify_deltas()


class Delta(IndexedExpr):
    def __init__(self, i1: Index, i2: Index):
        assert i1 != i2, f"Delta index({i1}) cannot be repeated"
        self.i1 = i1
        self.i2 = i2
        super().__init__()

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

    def _signless_diff(
        self, target: _TensorIndexing, dependencies: list[Tensor]
    ) -> tuple[list[Tensor], IndexedExpr]:
        indices_list = [self.i1, self.i2]
        _check_indices_diff(indices_list, target.indices)
        new_indices = indices_list + target.indices
        return dependencies, _make_indexed(_Zero, new_indices)

    def __str__(self) -> str:
        return self.add_sign(f"delta({self.i1}, {self.i2})", False)

    def get_children(self) -> list[IndexedExpr]:
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


class _TensorIndexing(IndexedExpr):
    tensor: _Tensor
    indices: list[Index]

    def __init__(
        self, tensor: _Tensor, indices: list[Index], sign: _Sign = _Sign.Plus
    ) -> None:
        print(tensor, indices)
        assert (
            len(indices) == tensor.rank
        ), f"Index count({len(indices)}) doesn't match rank({tensor.rank})"
        counts = _get_index_counts(indices)
        for i, c in counts.items():
            if c > 2:
                raise ValueError(f"Index {i} appears more the 2 times")
        self.tensor = tensor
        self.indices = indices
        super().__init__(sign)

    def get_free_indices(self) -> set[Index]:
        return _get_free_indices(self.indices)

    def get_dummy_indices(self) -> set[Index]:
        return _get_dummy_indices(self.indices)

    def _signless_diff(
        self, target: _TensorIndexing, dependencies: list[Tensor]
    ) -> tuple[list[Tensor], IndexedExpr]:
        _check_indices_diff(self.indices, target.indices)
        if target.tensor == self.tensor:
            if self.tensor.rank == 0:
                return dependencies, _ImplicitScalar(1)
            expr: IndexedExpr = Delta(self.indices[0], target.indices[0])
            for i, j in zip(self.indices[1:], target.indices[1:]):
                expr = expr * Delta(i, j)
            return dependencies, expr
        elif isinstance(self.tensor, Tensor) and self.tensor.value is not None:
            print(f"diff lhs {self.tensor} rhs {self.tensor.value.rhs}")
            dependencies, d_tensor = self.tensor.value._diff(
                self.tensor, target, dependencies
            )
            return dependencies, d_tensor[self.indices + target.indices]
        else:
            new_indices = self.indices + target.indices
            return dependencies, _make_indexed(_Zero, new_indices)

    def __str__(self) -> str:
        if self.indices:
            return self.add_sign(
                f"{self.tensor}[{",".join([str(i) for i in self.indices])}]", False
            )
        else:
            return self.add_sign(str(self.tensor), False)

    def get_children(self) -> list[IndexedExpr]:
        return []

    def replace_indices(self, replacements: dict[Index, Index]) -> None:
        for old, new in replacements.items():
            for n in range(len(self.indices)):
                if self.indices[n] == old:
                    self.indices[n] = new

    def simplify_deltas(self) -> None:
        return


class _Equality:
    def __init__(self, indices: list[Index], rhs: IndexedExpr) -> None:
        dummy = _get_dummy_indices(indices)
        assert not dummy, f"Dummy indices({dummy}) detected in assignment LHS"
        rhs_free_indices = rhs.get_free_indices()
        assert rhs_free_indices == set(
            indices
        ), f"RHS indices({rhs_free_indices}) don't match LHS({indices})"
        self.rhs = rhs
        self.free_indices = indices

    def _diff(
        self, y: Tensor, x: _TensorIndexing, dependencies: list[Tensor]
    ) -> tuple[list[Tensor], Tensor]:
        print(x, x.indices, self.rhs, self.rhs.get_dummy_indices(), self.rhs.get_free_indices())
        assert not (
            set(x.indices)
            & (self.rhs.get_dummy_indices() | self.rhs.get_free_indices())
        ), "Independent tensor cannot re-use indices"
        d_tensor = Tensor._derivative_tensor(y, len(x.indices))
        for dep in dependencies:
            # derivative already known
            if dep == y:
                return dependencies, d_tensor
        dependencies, expr = self.rhs._diff(x, dependencies)
        if Context.simplify_deltas:
            expr.simplify_deltas()
        d_indices = tuple(self.free_indices + x.indices)
        d_tensor[d_indices] = expr
        dependencies.append(d_tensor)
        return dependencies, d_tensor


def _make_indexed(T: Callable[[int], _Tensor], indices: list[Index]) -> IndexedExpr:
    return T(len(indices)).__getitem__(indices)


class _Tensor(ABC):
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def __getitem__(
        self, indices: Union[Index, Tuple[Index, ...], list[Index]]
    ) -> _TensorIndexing:
        indices_list = _listify_index(indices)
        return _TensorIndexing(self, indices_list)

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.__str__()


class Tensor(_Tensor):

    def __init__(self, rank: int, name: str) -> None:
        super().__init__(rank)
        _add_name(name, Tensor)
        self.name = name
        self.value: Optional[_Equality] = None
        self.derivative: Optional[Tensor] = None

    def _derivative_tensor(self, extra_rank: int) -> "Tensor":
        if self.derivative:
            return self.derivative
        else:
            new_name = f"d{self.name}"
            obj = Tensor(self.rank + extra_rank, new_name)
            self.derivative = obj
            return obj

    def __getitem__(
        self, indices: Union[Index, Tuple[Index, ...], list[Index]]
    ) -> _TensorIndexing:
        if self.value:
            indices_list = _listify_index(indices)
            indices_set = set(indices_list)
            # valid only if all indices belong to free ones, or indices are brand new
            rhs_dummy = self.value.rhs.get_dummy_indices()
            invalid_indices = indices_set & self.value.rhs.get_dummy_indices()
            assert (
                not invalid_indices
            ), f"Supplied indices({indices_set}) should be not be in rhs dummy indices set({rhs_dummy})"
        return super().__getitem__(indices)

    def __del__(self) -> None:
        _remove_name(self.name, Tensor)

    def __str__(self) -> str:
        return self.name

    def __setitem__(
        self, indices: Union[Index, Tuple[Index, ...]], expr: IndexedExpr
    ) -> None:
        indices_list = _listify_index(indices)
        assert len(indices_list) == self.rank, f"Tensor rank({self.rank}) doesn't match index count({len(indices_list)})"
        self.value = _Equality(indices_list, expr)

    def diff(self, x: _TensorIndexing | Scalar) -> list[Tensor]:
        if isinstance(x, Scalar):
            x = x.__getitem__(())
        assert isinstance(
            x.tensor, Tensor
        ), "Only subclasses of Tensor can be independent variables"
        assert not _get_dummy_indices(
            x.indices
        ), "Independent tensor cannot have dummy indices"
        assert self.value, f"Tensor({self}) must have an expression assigned"
        # tensor_indexed = self[self.value.free_indices]
        # statements = self._diff(x, [])
        # if Context.simplify_deltas:
        #     for s in statements:
        #         assert s.value, "Unreachable"
        #         s.value.rhs.simplify_deltas()
        return self.value._diff(self, x, [])[0]


class Scalar(Tensor):
    def __init__(self, name: str) -> None:
        super().__init__(0, name)

    def __mul__(self, other: "IndexedExpr" | int | float | Scalar) -> "IndexedExpr":
        return _promote_scalars(self) * _promote_scalars(other)

    def __rmul__(self, other: int | float) -> "IndexedExpr":
        return _promote_scalars(other) * _promote_scalars(self)

    def __add__(self, other: "IndexedExpr" | Scalar | int | float) -> "IndexedExpr":
        return _promote_scalars(self) + _promote_scalars(other)

    def __neg__(self) -> "IndexedExpr":
        return -_promote_scalars(self)

    def __sub__(self, other: "IndexedExpr" | Scalar | int | float) -> "IndexedExpr":
        other = -_promote_scalars(other)
        return _Sum.create(_promote_scalars(self), other)


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

    def __setitem__(
        self, indices: Union[Index, Tuple[Index, ...]], expr: IndexedExpr
    ) -> None:
        raise ValueError("0 tensor cannot be set")


def is_one(expr: IndexedExpr) -> bool:
    return isinstance(expr, _ImplicitScalar) and expr.num() == 1


def is_zero(expr: IndexedExpr) -> bool:
    return isinstance(expr, _TensorIndexing) and isinstance(expr.tensor, _Zero)


# TODO: Convert to _One if 1 is passed
class _ImplicitScalar(_TensorIndexing):
    class Tensor(_Tensor):
        def __init__(self, num: int | float) -> None:
            super().__init__(0)
            self.num = num

        def __str__(self) -> str:
            return str(self.num)

    def __init__(self, num: int | float) -> None:
        sign = _Sign.Plus
        if num < 0:
            num = -num
            sign = _Sign.Minus
        super().__init__(self.Tensor(num), [], sign)

    def num(self) -> int | float:
        assert isinstance(self.tensor, _ImplicitScalar.Tensor)
        return self.tensor.num

    def __setitem__(
        self, indices: Union[Index, Tuple[Index, ...]], expr: IndexedExpr
    ) -> None:
        raise ValueError("Implicit scalars cannot be set")
