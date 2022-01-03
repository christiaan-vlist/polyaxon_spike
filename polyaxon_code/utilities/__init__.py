"""A collection of isolated functions and classes which simplify our code base.

Attributes:
    click_types: Module with a collection of custom Click types
    const: Module with constants shared throughout the software

Functions:
    get_model_path: Retrieve the correct path for storing a model file
    safe_is_proper_subclass: Safely check if an object is a proper subclass
"""
from __future__ import annotations

import inspect
import typing

import cloudpathlib

from . import const


def safe_is_proper_subclass(
    thing: typing.Any,
    superclass: type | tuple[type, ...],
) -> bool:
    """Determine whether an object is proper subclass of a given type.

    This function differs from the built-in ``issubclass`` method as follows:
    - No errors are raised if a given value is not a class, and
    - A class does not count as a proper subclass of itself.

    Args:
        thing: The value to test
        superclass: A class or tuple of classes to compare ``thing`` against

    Returns:
        True if ``thing`` is a proper subclass of one of the superclasses

    Example:

        >>> class A: pass
        >>> class B(A): pass
        >>> class C(B): pass
        >>> class X: pass

        The function returns True for proper subclasses:

        >>> safe_is_proper_subclass(B, A)
        True
        >>> safe_is_proper_subclass(C, A)
        True

        And the function returns False otherwise:

        >>> safe_is_proper_subclass(X, A)
        False
        >>> safe_is_proper_subclass(A, B)
        False

        It also excludes classes as their own proper subclass:

        >>> safe_is_proper_subclass(A, A)
        False

        The function further returns False if a non-class parameter is given:

        >>> safe_is_proper_subclass(3, A)
        False
        >>> safe_is_proper_subclass(A(), B)
        False

        Finally, it also tests if it has one of multiple superclasses:

        >>> safe_is_proper_subclass(C, (A, B))
        True
        >>> safe_is_proper_subclass(B, (A, X))
        True
        >>> safe_is_proper_subclass(X, (A, B))
        False
    """
    if not inspect.isclass(thing):
        return False

    if issubclass(thing, superclass) and thing != superclass:
        return True

    return False
