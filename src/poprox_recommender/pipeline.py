from copy import deepcopy
from dataclasses import dataclass
from inspect import _empty, signature
from typing import Callable

from poprox_concepts import Article, ArticleSet, InterestProfile

StateValue = ArticleSet | InterestProfile
StateDict = dict[str, StateValue]


@dataclass
class ComponentSpec:
    component: Callable
    inputs: list[str]
    output: str


@dataclass
class PipelineState:
    elements: StateDict
    _last: str | None = None

    def __getitem__(self, key):
        return self.elements[key]

    def __setitem__(self, key, value):
        self.elements[key] = value

    @property
    def last(self) -> StateValue:
        return self.elements[self._last]

    @property
    def recs(self) -> list[Article]:
        return self.last.articles


class RecommendationPipeline:
    def __init__(self, name):
        self.name = name
        self.components = []
        self._state_types = {"candidate": ArticleSet, "clicked": ArticleSet, "profile": InterestProfile}

    def add(self, component: Callable, inputs: list[str], output: str):
        spec = ComponentSpec(component, inputs, output)
        self._validate_input_types(spec, self._state_types)
        self._state_types[output] = self._validate_output_type(spec)
        self.components.append(spec)

    def __call__(self, inputs: PipelineState | StateDict) -> PipelineState:
        # Avoid modifying the inputs
        state = deepcopy(inputs)
        if isinstance(state, dict):
            state = PipelineState(state)

        # Run each component in the order it was added
        for component_spec in self.components:
            state = self.run_component(component_spec, state)

        # Double check that we're returning the right type for recs
        if not isinstance(state.last, ArticleSet):
            msg = f"The final pipeline component must return ArticleSet, but received {type(state.recs)}"
            raise TypeError(msg)

        return state

    @property
    def last_output(self):
        return self.components[-1].output

    def run_component(self, component_spec: ComponentSpec, state: PipelineState):
        arguments = []
        for input_name in component_spec.inputs:
            arguments.append(state.elements[input_name])

        output = component_spec.component(*arguments)

        self._validate_return_type(component_spec, output)
        state._last = component_spec.output
        state[state._last] = output

        return state

    def _validate_input_types(self, spec: ComponentSpec, state_types: dict[str, type]):
        # Best effort type-checking to highlight when inputs are mismatched or out of order
        sig = signature(spec.component.__call__)
        sig_params = list(sig.parameters.values())

        for input_name, sig_param in zip(spec.inputs, sig_params):
            input_type = state_types[input_name]
            sig_param_type = sig_param.annotation
            if sig_param_type is not _empty and input_type != sig_param_type:
                msg = (
                    f"Component {spec.component} expected inputs with types {[p.annotation for p in sig_params]} "
                    f"but received inputs with types {[state_types[i] for i in spec.inputs]}"
                )

                raise TypeError(msg)

    def _validate_output_type(self, spec: ComponentSpec):
        # Best effort type-checking to highlight when output is
        # mismatched with existing state values types
        sig = signature(spec.component.__call__)
        output_name = spec.output

        state_type = self._state_types.get(output_name, None)
        output_type = sig.return_annotation

        if output_type is not _empty and output_type not in (ArticleSet, InterestProfile):
            msg = (
                f"Pipeline components must return ArticleSet or InterestProfile, "
                f"but received {type(output_type)} from {type(spec.component)}"
            )
            raise TypeError(msg)

        if state_type and output_type is not _empty and state_type != output_type:
            msg = (
                f"Component {spec.component} returns output with type {output_type} "
                f"but would need to return {state_type} in order to overwrite {output_name}"
            )

            raise TypeError(msg)

        return output_type

    def _validate_return_type(self, component_spec, output):
        expected_type = self._state_types.get(component_spec.output, None)
        if expected_type and not isinstance(output, expected_type):
            msg = (
                f"{type(component_spec.component)} is expected to return {expected_type}, "
                f"but received {type(output)}"
            )
            raise TypeError(msg)
