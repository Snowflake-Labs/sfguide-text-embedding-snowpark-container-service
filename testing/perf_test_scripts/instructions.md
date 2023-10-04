# Instructions

We don't use PyTest for performance testing, so these tests are just Python scripts.


## Requirements.

You need to `pip install aiohttp` to run these tests.

## Getting example text from wikipedia

Here's a useful "one"-liner example of getting plaintext extract from wikipedia.

```python
import requests
print(
    next(
        iter(
            requests.get("https://en.wikipedia.org/w/api.php?titles=Snowflake_Inc.&action=query&format=json&prop=extracts&explaintext=True")
            .json()
            ["query"]
            ["pages"]
            .values()
        )
    )
    ["extract"]
)
```
