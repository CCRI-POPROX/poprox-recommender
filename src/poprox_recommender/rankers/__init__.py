from typing import Any


class Ranker:
    def validate_algo_params(self, algo_params: dict[str, Any], valid_params: list[str]):
        for passed_param in algo_params.keys():
            if passed_param not in valid_params:
                msg = f"The parameter {passed_param} is not a valid parameter. Valid parameters are: {valid_params}"
                raise ValueError(msg)