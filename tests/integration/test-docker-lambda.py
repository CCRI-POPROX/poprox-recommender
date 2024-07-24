"""
Test the Lambda endpoint as run in a Docker container, encoding our request
as the AWS API gateway would.

Usage:
    test-docker-lambda.py [-v] [-p PORT] [-r FILE]

Args:
    -v, --verbose
        enable verbose log messages
    -p PORT, --port=PORT
        the port number where Docker is listening [default: 9000]
    -r FILE, --request=FILE
        load request payload from FILE [default: tests/basic-request.json]
"""

# metadata so this script can be run with 'pipx'
# /// script
# requires-python = ">= 3.10"
# dependencies = ["docopt>=0.6", "requests~=2.31"]
# ///

import json
import logging
import sys
from pathlib import Path

import requests
from docopt import docopt

logger = logging.getLogger("test-docker-lambda")


def main(args):
    log_level = logging.DEBUG if args["--verbose"] else logging.INFO
    logging.basicConfig(level=log_level, stream=sys.stderr)

    port = args["--port"]
    url = f"http://localhost:{port}/2015-03-31/functions/function/invocations"
    logger.info("lambda URL: %s", url)

    req_file = Path(args["--request"])
    logger.info("loading request from %s", req_file)
    req_txt = req_file.read_text()
    logger.debug("request content: %s", req_txt)

    logger.info("sending request")
    result = requests.post(url, json={"body": req_txt})

    logger.info("received response, code %s", result.status_code)
    logger.debug("response headers: %s", json.dumps(dict(result.headers), indent=2))
    logger.debug("response body: %s", json.dumps(result.json(), indent=2))

    if result.status_code != 200:
        logger.error("lambda request failed with code %s", result.status_code)
        logger.info("result: %s", result.text)
        raise AssertionError("HTTP request failed")

    res_data = result.json()
    if "errorType" in res_data:
        logger.error("Lambda function failed with type %s", res_data["errorType"])
        logger.info("Lambda error message: %s", res_data["errorMessage"])
        if res_data["stackTrace"]:
            logger.info("Stack trace:%s", [f"\n  {i}: {t}" for (i, t) in enumerate(res_data["stackTrace"], 1)])
        raise AssertionError("lambda function failed")

    if res_data["statusCode"] != 200:
        logger.error("HTTP succeeded but lambda failed with code %s", res_data["statusCode"])
        logger.info("result: %s", json.dumps(res_data, indent=2))
        raise AssertionError("lambda request failed")

    body = json.loads(res_data["body"])
    print(json.dumps(body, indent=2))
    if "recommendations" not in body:
        logger.error("result had no recommendations")
        raise AssertionError("missing recommendations")


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
