# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"Definition of the component interfaces."

# pyright: strict
from __future__ import annotations

import inspect
from abc import abstractmethod
from importlib import import_module
from types import FunctionType
from typing import Callable, ClassVar, Generic, ParamSpec, TypeAlias

from typing_extensions import Any, Protocol, Self, TypeVar, override, runtime_checkable

# COut is only return, so Component[U] can be assigned to Component[T] if U ≼ T.
P = ParamSpec("P")
COut = TypeVar("COut", covariant=True)
PipelineFunction: TypeAlias = Callable[..., COut]


@runtime_checkable
class Configurable(Protocol):  # pragma: nocover
    """
    Interface for configurable objects such as pipeline components with settings
    or hyperparameters.  A configurable object supports two operations:

    * saving its configuration with :meth:`get_config`.
    * creating a new instance from a saved configuration with the class method
      :meth:`from_config`.

    An object must implement both of these methods to be considered
    configurable.  Components extending the :class:`Component` class
    automatically have working versions of these methods if they define their
    constructor parameters and fields appropriately.

    .. note::

        Configuration data should be JSON-compatible (strings, numbers, etc.).
    """

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self:
        """
        Reinstantiate this component from configuration values.
        """
        raise NotImplementedError()

    def get_config(self) -> dict[str, object]:
        """
        Get this component's configured hyperparameters.
        """
        raise NotImplementedError()


class Component(Configurable, Generic[COut]):
    """
    Base class for pipeline component objects.  Any component that is not just a
    function should extend this class.

    Components are :class:`Configurable`.  The base class provides default
    implementations of :meth:`get_config` and :meth:`from_config` that inspect
    the constructor arguments and instance variables to automatically provide
    configuration support.  By default, all constructor parameters will be
    considered configuration parameters, and their values will be read from
    instance variables of the same name. Components can also define
    :data:`EXTRA_CONFIG_FIELDS` and :data:`IGNORED_CONFIG_FIELDS` class
    variables to modify this behavior. Missing attributes are silently ignored.

    To work as components, derived classes also need to implement a ``__call__``
    method to perform their operations.
    """

    EXTRA_CONFIG_FIELDS: ClassVar[list[str]] = []
    """
    Names of instance variables that should be included in the configuration
    dictionary even though they do not correspond to named constructor
    arguments.

    .. note::

        This is rarely needed, and usually needs to be coupled with ``**kwargs``
        in the constructor to make the resulting objects constructible.
    """

    IGNORED_CONFIG_FIELDS: ClassVar[list[str]] = []
    """
    Names of constructor parameters that should be excluded from the
    configuration dictionary.
    """

    @override
    def get_config(self) -> dict[str, object]:
        """
        Get the configuration by inspecting the constructor and instance
        variables.
        """
        sig = inspect.signature(self.__class__)
        names = list(sig.parameters.keys()) + self.EXTRA_CONFIG_FIELDS
        params: dict[str, Any] = {}
        for name in names:
            if name not in self.IGNORED_CONFIG_FIELDS and hasattr(self, name):
                params[name] = getattr(self, name)

        return params

    @override
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self:
        """
        Create a class from the specified construction.  Configuration elements
        are passed to the constructor as keywrod arguments.
        """
        return cls(**cfg)

    @abstractmethod
    def __call__(self, **kwargs: Any) -> COut:
        """
        Run the pipeline's operation and produce a result.  This is the key
        method for components to implement.

        .. note::

            Due to limitations of Python's type model, derived classes will have
            a type error overriding this method when using strict type checking,
            because it is very cumbersome (if not impossible) to propagate
            parameter names and types through to a base class.  The solution is
            to either use basic type checking for implementations, or to disable
            the typechecker on the ``__call__`` signature definition.
        """
        ...


def instantiate_component(comp: str | type | FunctionType, config: dict[str, Any] | None) -> Callable[..., object]:
    if isinstance(comp, str):
        mname, oname = comp.split(":", 1)
        mod = import_module(mname)
        comp = getattr(mod, oname)

    # make the type checker happy
    assert not isinstance(comp, str)

    if isinstance(comp, FunctionType):
        return comp
    elif issubclass(comp, Configurable):
        if config is None:
            config = {}
        return comp.from_config(config)  # type: ignore
    else:
        return comp()  # type: ignore
