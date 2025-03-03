import base64
import gzip
import json

import requests

url = "http://localhost:9000/2015-03-31/functions/function/invocations"

req_txt = json.dumps({"test": "value"})


event = {
    "resource": "/",
    "path": "/",
    "httpMethod": "POST",
    "headers": {"Content-Type": "application/json", "Content-Encoding": "gzip"},
    "requestContext": {},
    "body": base64.encodebytes(gzip.compress(req_txt.encode())).decode("ascii"),
    "queryStringParameters": {"pipeline": "nrms"},
    "isBase64Encoded": True,
}

result = requests.post(url, json=event)

print(result.content)
