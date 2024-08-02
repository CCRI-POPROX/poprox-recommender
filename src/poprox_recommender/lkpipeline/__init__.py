# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
A vendored copy of LensKit's pipeline abstraction, without trainability support.
"""

# pyright: strict
from __future__ import annotations

import logging
from types import FunctionType
from typing import Literal, cast
from uuid import uuid4

from typing_extensions import Any, LiteralString, TypeVar, overload

from .components import Component, ConfigurableComponent
from .nodes import ND, ComponentNode, FallbackNode, InputNode, LiteralNode, Node

__all__ = [
    "Pipeline",
    "Node",
    "Component",
    "ConfigurableComponent",
]

_log = logging.getLogger(__name__)

# common type var for quick use
T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")


class Pipeline:
    """
    LensKit recommendation pipeline.  This is the core abstraction for using
    LensKit models and other components to produce recommendations in a useful
    way.  It allows you to wire together components in (mostly) abitrary graphs,
    train them on data, and serialize pipelines to disk for use elsewhere.

    If you have a scoring model and just want to generate recommenations with a
    default setup and minimal configuration, see :func:`topn_pipeline`.
    """

    _nodes: dict[str, Node[Any]]
    _aliases: dict[str, Node[Any]]
    _defaults: dict[str, Node[Any] | Any]
    _components: dict[str, Component[Any]]

    def __init__(self):
        self._nodes = {}
        self._aliases = {}
        self._defaults = {}
        self._components = {}
        self._clear_caches()

    @property
    def nodes(self) -> list[Node[object]]:
        """
        Get the nodes in the pipeline graph.
        """
        return list(self._nodes.values())

    @overload
    def node(self, node: str, *, missing: Literal["error"] = "error") -> Node[object]: ...
    @overload
    def node(self, node: str, *, missing: Literal["none"] | None) -> Node[object] | None: ...
    @overload
    def node(self, node: Node[T]) -> Node[T]: ...
    def node(self, node: str | Node[Any], *, missing: Literal["error", "none"] | None = "error") -> Node[object] | None:
        """
        Get the pipeline node with the specified name.  If passed a node, it
        returns the node or fails if the node is not a member of the pipeline.

        Args:
            node:
                The name of the pipeline node to look up, or a node to check for
                membership.

        Returns:
            The pipeline node, if it exists.

        Raises:
            KeyError:
                The specified node does not exist.
        """
        if isinstance(node, Node):
            self._check_member_node(node)
            return node
        elif node in self._aliases:
            return self._aliases[node]
        elif node in self._nodes:
            return self._nodes[node]
        elif missing == "none" or missing is None:
            return None
        else:
            raise KeyError(f"node {node}")

    def create_input(self, name: str, *types: type[T] | None) -> Node[T]:
        """
        Create an input node for the pipeline.  Pipelines expect their inputs to
        be provided when they are run.

        Args:
            name:
                The name of the input.  The name must be unique in the pipeline
                (among both components and inputs).
            types:
                The allowable types of the input; input data can be of any
                specified type.  If ``None`` is among the allowed types, the
                input can be omitted.

        Returns:
            A pipeline node representing this input.

        Raises:
            ValueError:
                a node with the specified ``name`` already exists.
        """
        self._check_available_name(name)

        node = InputNode[Any](name, types=set((t if t is not None else type[None]) for t in types))
        self._nodes[name] = node
        self._clear_caches()
        return node

    def literal(self, value: T) -> LiteralNode[T]:
        name = str(uuid4())
        node = LiteralNode(name, value, types=set([type(value)]))
        self._nodes[name] = node
        return node

    def set_default(self, name: LiteralString, node: Node[Any] | object) -> None:
        """
        Set the default wiring for a component input.  Components that declare
        an input parameter with the specified ``name`` but no configured input
        will be wired to this node.

        This is intended to be used for things like wiring up `user` parameters
        to semi-automatically receive the target user's identity and history.

        Args:
            name:
                The name of the parameter to set a default for.
            node:
                The node or literal value to wire to this parameter.
        """
        if not isinstance(node, Node):
            node = self.literal(node)
        self._defaults[name] = node
        self._clear_caches()

    def get_default(self, name: str) -> Node[Any] | None:
        """
        Get the default wiring for an input name.
        """
        return self._defaults.get(name, None)

    def alias(self, alias: str, node: Node[Any] | str) -> None:
        """
        Create an alias for a node.  After aliasing, the node can be retrieved
        from :meth:`node` using either its original name or its alias.

        Args:
            alias:
                The alias to add to the node.
            node:
                The node (or node name) to alias.

        Raises:
            ValueError:
                if the alias is already used as an alias or node name.
        """
        node = self.node(node)
        self._check_available_name(alias)
        self._aliases[alias] = node
        self._clear_caches()

    def add_component(self, name: str, obj: Component[ND], **inputs: Node[Any] | object) -> Node[ND]:
        """
        Add a component and connect it into the graph.

        Args:
            name:
                The name of the component in the pipeline.  The name must be
                unique in the pipeline (among both components and inputs).
            obj:
                The component itself.
            inputs:
                The component's input wiring.  See :ref:`pipeline-connections`
                for details.

        Returns:
            The node representing this component in the pipeline.
        """
        self._check_available_name(name)

        node = ComponentNode(name, obj)
        self._nodes[name] = node
        self._components[name] = obj

        self.connect(node, **inputs)

        self._clear_caches()
        return node

    def replace_component(
        self,
        name: str | Node[ND],
        obj: Component[ND],
        **inputs: Node[Any] | object,
    ) -> Node[ND]:
        """
        Replace a component in the graph.  The new component must have a type
        that is compatible with the old component.  The old component's input
        connections will be replaced (as the new component may have different
        inputs), but any connections that use the old component to supply an
        input will use the new component instead.
        """
        if isinstance(name, Node):
            name = name.name

        node = ComponentNode(name, obj)
        self._nodes[name] = node
        self._components[name] = obj

        self.connect(node, **inputs)

        self._clear_caches()
        return node

    def use_first_of(self, name: str, *nodes: Node[T | None]) -> Node[T]:
        """
        Create a new node whose value is the first defined (not ``None``) value
        of the specified nodes.  If a node is an input node and its value is not
        supplied, it is treated as ``None`` in this case instead of failing the
        run. This method is used for things like filling in optional pipeline
        inputs.  For example, if you want the pipeline to take candidate items
        through an ``items`` input, but look them up from the user's history and
        the training data if ``items`` is not supplied, you would do:

        .. code:: python

            pipe = Pipeline()
            # allow candidate items to be optionally specified
            items = pipe.create_input('items', list[EntityId], None)
            # find candidates from the training data (optional)
            lookup_candidates = pipe.add_component(
                'select-candidates',
                UnratedTrainingItemsCandidateSelector(),
                user=history,
            )
            # if the client provided items as a pipeline input, use those; otherwise
            # use the candidate selector we just configured.
            candidates = pipe.use_first_of('candidates', items, lookup_candidates)

        .. note::

            This method does not distinguish between an input being unspecified and
            explicitly specified as ``None``.

        .. note::

            This method does *not* implement item-level fallbacks, only
            fallbacks at the level of entire results.  That is, you can use it
            to use component A as a fallback for B if B returns ``None``, but it
            will not use B to fill in missing scores for individual items that A
            did not score.  A specific itemwise fallback component is needed for
            such an operation.

        Args:
            name:
                The name of the node.
            nodes:
                The nodes to try, in order, to satisfy this node.
        """
        node = FallbackNode(name, list(nodes))
        self._nodes[name] = node
        self._clear_caches()
        return node

    def connect(self, obj: str | Node[Any], **inputs: Node[Any] | str | object):
        """
        Provide additional input connections for a component that has already
        been added.  See :ref:`pipeline-connections` for details.

        Args:
            obj:
                The name or node of the component to wire.
            inputs:
                The component's input wiring.  For each keyword argument in the
                component's function signature, that argument can be provided
                here with an input that the pipeline will provide to that
                argument of the component when the pipeline is run.
        """
        if isinstance(obj, Node):
            node = obj
        else:
            node = self.node(obj)
        if not isinstance(node, ComponentNode):
            raise TypeError(f"only component nodes can be wired, not {node}")

        for k, n in inputs.items():
            if isinstance(n, Node):
                n = cast(Node[Any], n)
                self._check_member_node(n)
                node.connections[k] = n.name
            else:
                lit = self.literal(n)
                node.connections[k] = lit.name

        self._clear_caches()

    def component_configs(self) -> dict[str, dict[str, Any]]:
        """
        Get the configurations for the components.  This is the configurations
        only, it does not include pipeline inputs or wiring.
        """
        return {
            name: comp.get_config()
            for (name, comp) in self._components.items()
            if isinstance(comp, ConfigurableComponent)
        }

    def clone(self, *, params: bool = False) -> Pipeline:
        """
        Clone the pipeline, optionally including trained parameters.

        Args:
            params:
                Pass ``True`` to clone parameters as well as the configuration
                and wiring.

        Returns:
            A new pipeline with the same components and wiring, but fresh
            instances created by round-tripping the configuration.
        """
        if params:  # pragma: nocover
            raise NotImplementedError()

        clone = Pipeline()
        for node in self.nodes:
            match node:
                case InputNode(name, types=types):
                    if types is None:
                        types = set[type]()
                    clone.create_input(name, *types)
                case LiteralNode(name, value):
                    clone._nodes[name] = LiteralNode(name, value)
                case FallbackNode(name, alts):
                    clone.use_first_of(name, *alts)
                case ComponentNode(name, comp, _inputs, wiring):
                    if isinstance(comp, FunctionType):
                        comp = comp
                    elif isinstance(comp, ConfigurableComponent):
                        comp = comp.__class__.from_config(comp.get_config())
                    else:
                        comp = comp.__class__()
                    cn = clone.add_component(node.name, comp)  # type: ignore
                    for wn, wt in wiring.items():
                        clone.connect(cn, **{wn: clone.node(wt)})
                case _:  # pragma: nocover
                    raise RuntimeError(f"invalid node {node}")

        return clone

    @overload
    def run(self, /, **kwargs: object) -> object: ...
    @overload
    def run(self, node: str, /, **kwargs: object) -> object: ...
    @overload
    def run(self, n1: str, n2: str, /, *nrest: str, **kwargs: object) -> tuple[object]: ...
    @overload
    def run(self, node: Node[T], /, **kwargs: object) -> T: ...
    @overload
    def run(self, n1: Node[T1], n2: Node[T2], /, **kwargs: object) -> tuple[T1, T2]: ...
    @overload
    def run(self, n1: Node[T1], n2: Node[T2], n3: Node[T3], /, **kwargs: object) -> tuple[T1, T2, T3]: ...
    @overload
    def run(
        self, n1: Node[T1], n2: Node[T2], n3: Node[T3], n4: Node[T4], /, **kwargs: object
    ) -> tuple[T1, T2, T3, T4]: ...
    @overload
    def run(
        self,
        n1: Node[T1],
        n2: Node[T2],
        n3: Node[T3],
        n4: Node[T4],
        n5: Node[T5],
        /,
        **kwargs: object,
    ) -> tuple[T1, T2, T3, T4, T5]: ...
    def run(self, *nodes: str | Node[Any], **kwargs: object) -> object:
        """
        Run the pipeline and obtain the return value(s) of one or more of its
        components.  See :ref:`pipeline-execution` for details of the pipeline
        execution model.

        .. todo::
            Add cycle detection.

        Args:
            nodes:
                The component(s) to run.
            kwargs:
                The pipeline's inputs, as defined with :meth:`create_input`.

        Returns:
            The pipeline result.  If zero or one nodes are specified, the result
            is returned as-is. If multiple nodes are specified, their results
            are returned in a tuple.

        Raises:
            ValueError:
                when one or more required inputs are missing.
            TypeError:
                when one or more required inputs has an incompatible type.
            other:
                exceptions thrown by components are passed through.
        """
        from .runner import PipelineRunner

        runner = PipelineRunner(self, kwargs)
        if not nodes:
            nodes = (self._last_node(),)
        results = [runner.run(self.node(n)) for n in nodes]

        if len(results) > 1:
            return tuple(results)
        else:
            return results[0]

    def _last_node(self) -> Node[object]:
        if not self._nodes:
            raise RuntimeError("pipeline is empty")
        return list(self._nodes.values())[-1]

    def _check_available_name(self, name: str) -> None:
        if name in self._nodes or name in self._aliases:
            raise ValueError(f"pipeline already has node {name}")

    def _check_member_node(self, node: Node[Any]) -> None:
        nw = self._nodes.get(node.name)
        if nw is not node:
            raise RuntimeError(f"node {node} not in pipeline")

    def _clear_caches(self):
        pass
