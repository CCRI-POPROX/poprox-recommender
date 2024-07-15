from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from poprox_concepts import ArticleSet, InterestProfile

PipelineState = dict[str, ArticleSet | InterestProfile]


@dataclass
class ComponentSpec:
    component: Callable
    inputs: list[str]
    output: str


class RecommendationPipeline:
    def __init__(self, name):
        self.name = name
        self.components = []

    def add(self, component: Callable, inputs: list[str], output: str):
        self.components.append(ComponentSpec(component, inputs, output))

    def __call__(self, inputs: PipelineState) -> PipelineState:
        # Avoid modifying the inputs
        state = deepcopy(inputs)

        # Run each component in the order it was added
        for component_spec in self.components:
            state = self.run_component(component_spec, state)

        recs = state[self.last_output]

        # Double check that we're returning the right type for recs
        if not isinstance(recs, ArticleSet):
            msg = f"The final pipeline component must return ArticleSet, but received {type(recs)}"
            raise TypeError(msg)

        return state

    @property
    def last_output(self):
        return self.components[-1].output

    def run_component(self, component_spec: ComponentSpec, state: dict[str, ArticleSet | InterestProfile]):
        arguments = []
        for input_name in component_spec.inputs:
            arguments.append(state[input_name])

        output = component_spec.component(*arguments)

        if not isinstance(output, (ArticleSet, InterestProfile)):
            msg = (
                f"Pipeline components must return ArticleSet or InterestProfile, "
                f"but received {type(output)} from {type(component_spec.component)}"
            )
            raise TypeError(msg)

        state[component_spec.output] = output

        return state
