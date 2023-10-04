from base64 import b64decode
from typing import Any
from typing import Dict
from typing import Sequence

import requests

ENDPOINT = "http://localhost:8000/embed"
# Some random UUID4s.
MOCK_QUERY_ID = "b34ae2f3-e954-4243-b23a-7554de390307"
MOCK_QUERY_BATCH_ID = "fc28d96a-de2c-490b-ba21-458c8786efee"
MOCK_HEADERS = {
    "sf-external-function-current-query-id": MOCK_QUERY_ID,
    "sf-external-function-query-batch-id": MOCK_QUERY_BATCH_ID,
}


def run_batch(texts: Sequence[str]) -> Dict[Any, Any]:
    json_payload = {"data": [[i, text] for i, text in enumerate(texts)]}
    result = requests.post(ENDPOINT, json=json_payload, headers=MOCK_HEADERS)
    assert result.status_code == 200
    result_json = result.json()
    return result_json


def test_hello_world():
    result_json = run_batch(["Hello, world!"])
    assert "data" in result_json
    data = result_json["data"]
    assert len(data) == 1
    row = data[0]
    assert len(row) == 2
    row_id, row_content = row
    assert row_id == 0
    content_bytes = b64decode(row_content)
    assert len(content_bytes) > 0
    # Check that the contents' byte length is consistent with 4-byte wide float32
    assert len(content_bytes) % 4 == 0, "Content should be float32 objects"
