from copy import deepcopy
from dataclasses import dataclass
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

    def add(self, component: Callable, inputs: list[str], output: str):
        self.components.append(ComponentSpec(component, inputs, output))

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

        if not isinstance(output, (ArticleSet, InterestProfile)):
            msg = (
                f"Pipeline components must return ArticleSet or InterestProfile, "
                f"but received {type(output)} from {type(component_spec.component)}"
            )
            raise TypeError(msg)

        state._last = component_spec.output
        state[state._last] = output

        return state
