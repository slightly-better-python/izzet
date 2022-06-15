# Python 3.8+
import collections
import gc
import inspect
import sys
from collections import OrderedDict
from collections.abc import Sequence
from enum import Enum
from inspect import isgeneratorfunction
from io import BufferedIOBase
from io import IOBase
from io import RawIOBase
from io import TextIOBase
from typing import AbstractSet
from typing import Any
from typing import AsyncIterable
from typing import AsyncIterator
from typing import BinaryIO
from typing import Callable
from typing import Dict
from typing import Generator
from typing import get_type_hints
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NewType
from typing import Optional
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from unittest.mock import Mock
from warnings import warn
from weakref import WeakKeyDictionary
from weakref import WeakValueDictionary

try:
    from typing_extensions import Literal
except ImportError:
    try:
        from typing import Literal
    except ImportError:
        Literal = None

# Python 3.5.4+ / 3.6.2+
try:
    from typing_extensions import NoReturn
except ImportError:
    try:
        from typing import NoReturn
    except ImportError:
        NoReturn = None

# Python 3.6+
try:
    from inspect import isasyncgen, isasyncgenfunction
    from typing import AsyncGenerator
except ImportError:
    AsyncGenerator = None


    def isasyncgen(obj):
        return False


    def isasyncgenfunction(func):
        return False

# Python 3.8+
try:
    from typing import ForwardRef

    evaluate_forwardref = ForwardRef._evaluate
except ImportError:
    from typing import _ForwardRef as ForwardRef

    evaluate_forwardref = ForwardRef._eval_type

if sys.version_info >= (3, 10):
    pass
else:
    _typed_dict_meta_types = ()
    if sys.version_info >= (3, 8):
        from typing import _TypedDictMeta

        _typed_dict_meta_types += (_TypedDictMeta,)

    try:
        from typing_extensions import _TypedDictMeta

        _typed_dict_meta_types += (_TypedDictMeta,)
    except ImportError:
        pass


    def is_typeddict(tp) -> bool:
        return isinstance(tp, _typed_dict_meta_types)

if TYPE_CHECKING:
    _F = TypeVar("_F")


    def typeguard_ignore(f: _F) -> _F:
        """This decorator is a noop during static type-checking."""
        return f
else:
    pass

_type_hints_map = WeakKeyDictionary()  # type: Dict[FunctionType, Dict[str, Any]]
_functions_map = WeakValueDictionary()  # type: Dict[CodeType, FunctionType]
_missing = object()

T_CallableOrType = TypeVar('T_CallableOrType', bound=Callable[..., Any])

# Lifted from mypy.sharedparse
BINARY_MAGIC_METHODS = {
    "__add__",
    "__and__",
    "__cmp__",
    "__divmod__",
    "__div__",
    "__eq__",
    "__floordiv__",
    "__ge__",
    "__gt__",
    "__iadd__",
    "__iand__",
    "__idiv__",
    "__ifloordiv__",
    "__ilshift__",
    "__imatmul__",
    "__imod__",
    "__imul__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__itruediv__",
    "__ixor__",
    "__le__",
    "__lshift__",
    "__lt__",
    "__matmul__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__or__",
    "__pow__",
    "__radd__",
    "__rand__",
    "__rdiv__",
    "__rfloordiv__",
    "__rlshift__",
    "__rmatmul__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__rpow__",
    "__rrshift__",
    "__rshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
    "__sub__",
    "__truediv__",
    "__xor__",
}


class ForwardRefPolicy(Enum):
    """Defines how unresolved forward references are handled."""

    ERROR = 1  #: propagate the :exc:`NameError` from :func:`~typing.get_type_hints`
    WARN = 2  #: remove the annotation and emit a TypeHintWarning
    #: replace the annotation with the argument's class if the qualified name matches, else remove
    #: the annotation
    GUESS = 3


class TypeHintWarning(UserWarning):
    """
    A warning that is emitted when a type hint in string form could not be resolved to an actual
    type.
    """


class TypeCheckMemo:
    __slots__ = 'globals', 'locals'

    def __init__(self, globals: Dict[str, Any], locals: Dict[str, Any]):
        self.globals = globals
        self.locals = locals


def _strip_annotation(annotation):
    if isinstance(annotation, str):
        return annotation.strip("'")
    else:
        return annotation


class _CallMemo(TypeCheckMemo):
    __slots__ = 'func', 'func_name', 'arguments', 'is_generator', 'type_hints'

    def __init__(self, func: Callable, frame_locals: Optional[Dict[str, Any]] = None,
                 args: tuple = None, kwargs: Dict[str, Any] = None,
                 forward_refs_policy=ForwardRefPolicy.ERROR):
        super().__init__(func.__globals__, frame_locals)
        self.func = func
        self.func_name = function_name(func)
        self.is_generator = isgeneratorfunction(func)
        signature = inspect.signature(func)

        if args is not None and kwargs is not None:
            self.arguments = signature.bind(*args, **kwargs).arguments
        else:
            assert frame_locals is not None, 'frame must be specified if args or kwargs is None'
            self.arguments = frame_locals

        self.type_hints = _type_hints_map.get(func)
        if self.type_hints is None:
            while True:
                if sys.version_info < (3, 5, 3):
                    frame_locals = dict(frame_locals)

                try:
                    hints = get_type_hints(func, localns=frame_locals)
                except NameError as exc:
                    if forward_refs_policy is ForwardRefPolicy.ERROR:
                        raise

                    typename = str(exc).split("'", 2)[1]
                    for param in signature.parameters.values():
                        if _strip_annotation(param.annotation) == typename:
                            break
                    else:
                        raise

                    func_name = function_name(func)
                    if forward_refs_policy is ForwardRefPolicy.GUESS:
                        if param.name in self.arguments:
                            argtype = self.arguments[param.name].__class__
                            stripped = _strip_annotation(param.annotation)
                            if stripped == argtype.__qualname__:
                                func.__annotations__[param.name] = argtype
                                msg = ('Replaced forward declaration {!r} in {} with {!r}'
                                       .format(stripped, func_name, argtype))
                                warn(TypeHintWarning(msg))
                                continue

                    msg = 'Could not resolve type hint {!r} on {}: {}'.format(
                        param.annotation, function_name(func), exc)
                    warn(TypeHintWarning(msg))
                    del func.__annotations__[param.name]
                else:
                    break

            self.type_hints = OrderedDict()
            for name, parameter in signature.parameters.items():
                if name in hints:
                    annotated_type = hints[name]

                    # PEP 428 discourages it by MyPy does not complain
                    if parameter.default is None:
                        annotated_type = Optional[annotated_type]

                    if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                        self.type_hints[name] = Tuple[annotated_type, ...]
                    elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                        self.type_hints[name] = Dict[str, annotated_type]
                    else:
                        self.type_hints[name] = annotated_type

            if 'return' in hints:
                self.type_hints['return'] = hints['return']

            _type_hints_map[func] = self.type_hints


def resolve_forwardref(maybe_ref, memo: TypeCheckMemo):
    if isinstance(maybe_ref, ForwardRef):
        if sys.version_info < (3, 9, 0):
            return evaluate_forwardref(maybe_ref, memo.globals, memo.locals)
        else:
            return evaluate_forwardref(maybe_ref, memo.globals, memo.locals, frozenset())

    else:
        return maybe_ref


def get_type_name(type_):
    name = (getattr(type_, '__name__', None) or getattr(type_, '_name', None) or
            getattr(type_, '__forward_arg__', None))
    if name is None:
        origin = getattr(type_, '__origin__', None)
        name = getattr(origin, '_name', None)
        if name is None and not inspect.isclass(type_):
            name = type_.__class__.__name__.strip('_')

    args = getattr(type_, '__args__', ()) or getattr(type_, '__values__', ())
    if args != getattr(type_, '__parameters__', ()):
        if name == 'Literal':
            formatted_args = ', '.join(str(arg) for arg in args)
        else:
            formatted_args = ', '.join(get_type_name(arg) for arg in args)

        name = '{}[{}]'.format(name, formatted_args)

    module = getattr(type_, '__module__', None)
    if module not in (None, 'typing', 'typing_extensions', 'builtins'):
        name = module + '.' + name

    return name


def find_function(frame) -> Optional[Callable]:
    """
    Return a function object from the garbage collector that matches the frame's code object.

    This process is unreliable as several function objects could use the same code object.
    Fortunately the likelihood of this happening with the combination of the function objects
    having different type annotations is a very rare occurrence.

    :param frame: a frame object
    :return: a function object if one was found, ``None`` if not

    """
    func = _functions_map.get(frame.f_code)
    if func is None:
        for obj in gc.get_referrers(frame.f_code):
            if inspect.isfunction(obj):
                if func is None:
                    # The first match was found
                    func = obj
                else:
                    # A second match was found
                    return None

        # Cache the result for future lookups
        if func is not None:
            _functions_map[frame.f_code] = func
        else:
            raise LookupError('target function not found')

    return func


def qualified_name(obj) -> str:
    """
    Return the qualified name (e.g. package.module.Type) for the given object.

    Builtins and types from the :mod:`typing` package get special treatment by having the module
    name stripped from the generated name.

    """
    type_ = obj if inspect.isclass(obj) else type(obj)
    module = type_.__module__
    qualname = type_.__qualname__
    return qualname if module in ('typing', 'builtins') else '{}.{}'.format(module, qualname)


def function_name(func: Callable) -> str:
    """
    Return the qualified name of the given function.

    Builtins and types from the :mod:`typing` package get special treatment by having the module
    name stripped from the generated name.

    """
    # For partial functions and objects with __call__ defined, __qualname__ does not exist
    # For functions run in `exec` with a custom namespace, __module__ can be None
    module = getattr(func, '__module__', '') or ''
    qualname = (module + '.') if module not in ('builtins', '') else ''
    return qualname + getattr(func, '__qualname__', repr(func))


def check_callable(value, expected_type, memo: TypeCheckMemo) -> None:
    if not callable(value):
        raise TypeError('{} must be a callable'.format('argument'))

    if getattr(expected_type, "__args__", None):
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            return

        if hasattr(expected_type, '__result__'):
            # Python 3.5
            argument_types = expected_type.__args__
            check_args = argument_types is not Ellipsis
        else:
            # Python 3.6
            argument_types = expected_type.__args__[:-1]
            check_args = argument_types != (Ellipsis,)

        if check_args:
            # The callable must not have keyword-only arguments without defaults
            unfulfilled_kwonlyargs = [
                param.name for param in signature.parameters.values() if
                param.kind == inspect.Parameter.KEYWORD_ONLY and param.default == inspect.Parameter.empty]
            if unfulfilled_kwonlyargs:
                raise TypeError(
                    'callable passed as {} has mandatory keyword-only arguments in its '
                    'declaration: {}'.format('argument', ', '.join(unfulfilled_kwonlyargs)))

            num_mandatory_args = len([
                param.name for param in signature.parameters.values()
                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and
                   param.default is inspect.Parameter.empty])
            has_varargs = any(param for param in signature.parameters.values()
                              if param.kind == inspect.Parameter.VAR_POSITIONAL)

            if num_mandatory_args > len(argument_types):
                raise TypeError(
                    'callable passed as {} has too many arguments in its declaration; expected {} '
                    'but {} argument(s) declared'.format('argument', len(argument_types),
                                                         num_mandatory_args))
            elif not has_varargs and num_mandatory_args < len(argument_types):
                raise TypeError(
                    'callable passed as {} has too few arguments in its declaration; expected {} '
                    'but {} argument(s) declared'.format('argument', len(argument_types),
                                                         num_mandatory_args))


def check_dict(value, expected_type, memo: TypeCheckMemo) -> None:
    if not isinstance(value, dict):
        raise TypeError('type of {} must be a dict; got {} instead'.
                        format('argument', qualified_name(value)))

    if expected_type is not dict:
        if (hasattr(expected_type, "__args__") and
                expected_type.__args__ not in (None, expected_type.__parameters__)):
            key_type, value_type = expected_type.__args__
            if key_type is not Any or value_type is not Any:
                for k, v in value.items():
                    _check_type(k, key_type, memo)
                    _check_type(v, value_type, memo)


def check_typed_dict(value, expected_type, memo: TypeCheckMemo) -> None:
    declared_keys = frozenset(expected_type.__annotations__)
    if hasattr(expected_type, '__required_keys__'):
        required_keys = expected_type.__required_keys__
    else:  # py3.8 and lower
        required_keys = declared_keys if expected_type.__total__ else frozenset()

    existing_keys = frozenset(value)
    extra_keys = existing_keys - declared_keys
    if extra_keys:
        keys_formatted = ', '.join('"{}"'.format(key) for key in sorted(extra_keys))
        raise TypeError('extra key(s) ({}) in {}'.format(keys_formatted, 'argument'))

    missing_keys = required_keys - existing_keys
    if missing_keys:
        keys_formatted = ', '.join('"{}"'.format(key) for key in sorted(missing_keys))
        raise TypeError('required key(s) ({}) missing from {}'.format(keys_formatted, 'argument'))

    for key, argtype in get_type_hints(expected_type).items():
        argvalue = value.get(key, _missing)
        if argvalue is not _missing:
            _check_type(argvalue, argtype, memo)


def check_list(value, expected_type, memo: TypeCheckMemo) -> None:
    if not isinstance(value, list):
        raise TypeError('type of {} must be a list; got {} instead'.
                        format('argument', qualified_name(value)))

    if expected_type is not list:
        if hasattr(expected_type, "__args__") and expected_type.__args__ not in \
                (None, expected_type.__parameters__):
            value_type = expected_type.__args__[0]
            if value_type is not Any:
                for i, v in enumerate(value):
                    _check_type(v, value_type, memo)


def check_sequence(value, expected_type, memo: TypeCheckMemo) -> None:
    if not isinstance(value, Sequence):
        raise TypeError('type of {} must be a sequence; got {} instead'.
                        format('argument', qualified_name(value)))

    if hasattr(expected_type, "__args__") and expected_type.__args__ not in \
            (None, expected_type.__parameters__):
        value_type = expected_type.__args__[0]
        if value_type is not Any:
            for i, v in enumerate(value):
                _check_type(v, value_type, memo)


def check_set(value, expected_type, memo: TypeCheckMemo) -> None:
    if not isinstance(value, AbstractSet):
        raise TypeError('type of {} must be a set; got {} instead'.
                        format('argument', qualified_name(value)))

    if expected_type is not set:
        if hasattr(expected_type, "__args__") and expected_type.__args__ not in \
                (None, expected_type.__parameters__):
            value_type = expected_type.__args__[0]
            if value_type is not Any:
                for v in value:
                    _check_type(v, value_type, memo)


def check_tuple(value, expected_type, memo: TypeCheckMemo) -> None:
    # Specialized check for NamedTuples
    is_named_tuple = False
    if sys.version_info < (3, 8, 0):
        is_named_tuple = hasattr(expected_type, '_field_types')  # deprecated since python 3.8
    else:
        is_named_tuple = hasattr(expected_type, '__annotations__')

    if is_named_tuple:
        if not isinstance(value, expected_type):
            raise TypeError('must be a named tuple of type {}; got {} instead'.
                            format(qualified_name(expected_type), qualified_name(value)))

        if sys.version_info < (3, 8, 0):
            field_types = expected_type._field_types
        else:
            field_types = expected_type.__annotations__

        for name, field_type in field_types.items():
            _check_type(getattr(value, name), field_type, memo)

        return
    elif not isinstance(value, tuple):
        raise TypeError('Must be a tuple; got {} instead'.
                        format(qualified_name(value)))

    if getattr(expected_type, '__tuple_params__', None):
        # Python 3.5
        use_ellipsis = expected_type.__tuple_use_ellipsis__
        tuple_params = expected_type.__tuple_params__
    elif getattr(expected_type, '__args__', None):
        # Python 3.6+
        use_ellipsis = expected_type.__args__[-1] is Ellipsis
        tuple_params = expected_type.__args__[:-1 if use_ellipsis else None]
    else:
        # Unparametrized Tuple or plain tuple
        return

    if use_ellipsis:
        element_type = tuple_params[0]
        for i, element in enumerate(value):
            _check_type(element, element_type, memo)
    elif tuple_params == ((),):
        if value != ():
            raise TypeError('{} is not an empty tuple but one was expected'.format('argument'))
    else:
        if len(value) != len(tuple_params):
            raise TypeError('{} has wrong number of elements (expected {}, got {} instead)'
                            .format('argument', len(tuple_params), len(value)))

        for i, (element, element_type) in enumerate(zip(value, tuple_params)):
            _check_type(element, element_type, memo)


def check_union(value, expected_type, memo: TypeCheckMemo) -> None:
    if hasattr(expected_type, '__union_params__'):
        # Python 3.5
        union_params = expected_type.__union_params__
    else:
        # Python 3.6+
        union_params = expected_type.__args__

    for type_ in union_params:
        try:
            _check_type(value, type_, memo)
            return
        except TypeError:
            pass

    typelist = ', '.join(get_type_name(t) for t in union_params)
    raise TypeError('type of {} must be one of ({}); got {} instead'.
                    format('argument', typelist, qualified_name(value)))


def check_class(value, expected_type, memo: TypeCheckMemo) -> None:
    if not inspect.isclass(value):
        raise TypeError('type of {} must be a type; got {} instead'.format(
            'argument', qualified_name(value)))

    # Needed on Python 3.7+
    if expected_type is Type:
        return

    if getattr(expected_type, '__origin__', None) in (Type, type):
        expected_class = expected_type.__args__[0]
    else:
        expected_class = expected_type

    if expected_class is Any:
        return
    elif isinstance(expected_class, TypeVar):
        check_typevar(value, expected_class, memo, True)
    elif getattr(expected_class, '__origin__', None) is Union:
        for arg in expected_class.__args__:
            try:
                check_class(value, arg, memo)
                break
            except TypeError:
                pass
        else:
            formatted_args = ', '.join(get_type_name(arg) for arg in expected_class.__args__)
            raise TypeError('{} must match one of the following: ({}); got {} instead'.format(
                'argument', formatted_args, qualified_name(value)
            ))
    elif not issubclass(value, expected_class):
        raise TypeError('{} must be a subclass of {}; got {} instead'.format(
            'argument', qualified_name(expected_class), qualified_name(value)))


def check_typevar(value, typevar: TypeVar, memo: TypeCheckMemo,
                  subclass_check: bool = False) -> None:
    value_type = value if subclass_check else type(value)
    subject = 'argument' if subclass_check else 'type of ' + 'argument'

    if typevar.__bound__ is not None:
        bound_type = resolve_forwardref(typevar.__bound__, memo)
        if not issubclass(value_type, bound_type):
            raise TypeError(
                '{} must be {} or one of its subclasses; got {} instead'
                    .format(subject, qualified_name(bound_type), qualified_name(value_type)))
    elif typevar.__constraints__:
        constraints = [resolve_forwardref(c, memo) for c in typevar.__constraints__]
        for constraint in constraints:
            try:
                _check_type(value, constraint, memo)
            except TypeError:
                pass
            else:
                break
        else:
            formatted_constraints = ', '.join(get_type_name(constraint)
                                              for constraint in constraints)
            raise TypeError('{} must match one of the constraints ({}); got {} instead'
                            .format(subject, formatted_constraints, qualified_name(value_type)))


def check_literal(value, expected_type, memo: TypeCheckMemo):
    def get_args(literal):
        try:
            args = literal.__args__
        except AttributeError:
            # Instance of Literal from typing_extensions
            args = literal.__values__

        retval = []
        for arg in args:
            if isinstance(arg, Literal.__class__) or getattr(arg, '__origin__', None) is Literal:
                # The first check works on py3.6 and lower, the second one on py3.7+
                retval.extend(get_args(arg))
            elif isinstance(arg, (int, str, bytes, bool, type(None), Enum)):
                retval.append(arg)
            else:
                raise TypeError('Illegal literal value: {}'.format(arg))

        return retval

    final_args = tuple(get_args(expected_type))
    if value not in final_args:
        raise TypeError('the value of {} must be one of {}; got {} instead'.
                        format('argument', final_args, value))


def check_number(value, expected_type):
    if expected_type is complex and not isinstance(value, (complex, float, int)):
        raise TypeError('type of {} must be either complex, float or int; got {} instead'.
                        format('argument', qualified_name(value.__class__)))
    elif expected_type is float and not isinstance(value, (float, int)):
        raise TypeError('type of {} must be either float or int; got {} instead'.
                        format('argument', qualified_name(value.__class__)))


def check_io(value, expected_type):
    if expected_type is TextIO:
        if not isinstance(value, TextIOBase):
            raise TypeError('type of {} must be a text based I/O object; got {} instead'.
                            format('argument', qualified_name(value.__class__)))
    elif expected_type is BinaryIO:
        if not isinstance(value, (RawIOBase, BufferedIOBase)):
            raise TypeError('type of {} must be a binary I/O object; got {} instead'.
                            format('argument', qualified_name(value.__class__)))
    elif not isinstance(value, IOBase):
        raise TypeError('type of {} must be an I/O object; got {} instead'.
                        format('argument', qualified_name(value.__class__)))


def check_protocol(value, expected_type):
    # TODO: implement proper compatibility checking and support non-runtime protocols
    if getattr(expected_type, '_is_runtime_protocol', False):
        if not isinstance(value, expected_type):
            raise TypeError('type of {} ({}) is not compatible with the {} protocol'.
                            format('argument', type(value).__qualname__, expected_type.__qualname__))


# Equality checks are applied to these
origin_type_checkers = {
    AbstractSet: check_set,
    Callable: check_callable,
    collections.abc.Callable: check_callable,
    dict: check_dict,
    Dict: check_dict,
    list: check_list,
    List: check_list,
    Sequence: check_sequence,
    collections.abc.Sequence: check_sequence,
    collections.abc.Set: check_set,
    set: check_set,
    Set: check_set,
    tuple: check_tuple,
    Tuple: check_tuple,
    type: check_class,
    Type: check_class,
    Union: check_union
}
_subclass_check_unions = hasattr(Union, '__union_set_params__')
if Literal is not None:
    origin_type_checkers[Literal] = check_literal

generator_origin_types = (Generator, collections.abc.Generator,
                          Iterator, collections.abc.Iterator,
                          Iterable, collections.abc.Iterable)
asyncgen_origin_types = (AsyncIterator, collections.abc.AsyncIterator,
                         AsyncIterable, collections.abc.AsyncIterable)
if AsyncGenerator is not None:
    asyncgen_origin_types += (AsyncGenerator,)
if hasattr(collections.abc, 'AsyncGenerator'):
    asyncgen_origin_types += (collections.abc.AsyncGenerator,)


def _check_type(value, expected_type, memo: Optional[TypeCheckMemo] = None, *,
                _globals: Optional[Dict[str, Any]] = None,
                _locals: Optional[Dict[str, Any]] = None) -> None:
    """
    Ensure that ``value`` matches ``expected_type``.

    The types from the :mod:`typing` module do not support :func:`isinstance` or :func:`issubclass`
    so a number of type specific checks are required. This function knows which checker to call
    for which type.

    :param value: value to be checked against ``expected_type``
    :param expected_type: a class or generic type instance
    :param _globals: dictionary of global variables to use for resolving forward references
        (defaults to the calling frame's globals)
    :param _locals: dictionary of local variables to use for resolving forward references
        (defaults to the calling frame's locals)
    :raises TypeError: if there is a type mismatch

    """

    if expected_type is Any or isinstance(value, Mock):
        return

    if expected_type is None:
        # Only happens on < 3.6
        expected_type = type(None)

    expected_type = resolve_forwardref(expected_type, memo)
    origin_type = getattr(expected_type, '__origin__', None)
    if origin_type is not None:
        checker_func = origin_type_checkers.get(origin_type)
        if checker_func:
            checker_func(value, expected_type, memo)
        else:
            _check_type(value, origin_type, memo)
    elif inspect.isclass(expected_type):
        if issubclass(expected_type, Tuple):
            check_tuple(value, expected_type, memo)
        elif issubclass(expected_type, (float, complex)):
            check_number(value, expected_type)
        elif _subclass_check_unions and issubclass(expected_type, Union):  # noqa
            check_union(value, expected_type, memo)
        elif isinstance(expected_type, TypeVar):
            check_typevar(value, expected_type, memo)
        elif issubclass(expected_type, IO):
            check_io(value, expected_type)
        elif is_typeddict(expected_type):
            check_typed_dict(value, expected_type, memo)
        elif getattr(expected_type, '_is_protocol', False):
            check_protocol(value, expected_type)
        else:
            expected_type = (getattr(expected_type, '__extra__', None) or origin_type or
                             expected_type)

            if expected_type is bytes:
                # As per https://github.com/python/typing/issues/552
                if not isinstance(value, (bytearray, bytes, memoryview)):
                    raise TypeError('type of {} must be bytes-like; got {} instead'
                                    .format('argument', qualified_name(value)))
            elif not isinstance(value, expected_type):
                raise TypeError(
                    'type of {} must be {}; got {} instead'
                        .format('argument', qualified_name(expected_type), qualified_name(value)))
    elif isinstance(expected_type, TypeVar):
        # Only happens on < 3.6
        check_typevar(value, expected_type, memo)
    elif isinstance(expected_type, Literal.__class__):
        # Only happens on < 3.7 when using Literal from typing_extensions
        check_literal(value, expected_type, memo)
    elif expected_type.__class__ is NewType:
        # typing.NewType on Python 3.10+
        return _check_type(value, expected_type.__supertype__, memo)
    elif (inspect.isfunction(expected_type) and
          getattr(expected_type, "__module__", None) == "typing" and
          getattr(expected_type, "__qualname__", None).startswith("NewType.") and
          hasattr(expected_type, "__supertype__")):
        # typing.NewType on Python 3.9 and below
        return _check_type(value, expected_type.__supertype__, memo)
