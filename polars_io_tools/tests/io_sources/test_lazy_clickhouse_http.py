"""Exercise HTTP decoding without replacing the ClickHouse batch reader."""

import gzip
import io
import zlib

import polars as pl
import pyarrow as pa
import pytest
import requests
from polars.testing import assert_frame_equal
from urllib3.response import HTTPResponse

import polars_io_tools.io_sources.lazy_clickhouse_reader as ch


@pytest.mark.parametrize("content_encoding", [None, "gzip", "deflate"])
@pytest.mark.parametrize("empty", [False, True])
def test_scan_clickhouse_decodes_http_response(monkeypatch, content_encoding, empty):
    expected = pl.DataFrame({"id": [1, 2, 3, 4], "label": ["alpha", "β", None, ""]})
    if empty:
        expected = expected.head(0)
    queries = []
    responses = []

    def post(url, *, params, stream):
        assert stream is True
        query = params["query"]
        queries.append(query)
        table = expected.head(0).to_arrow() if "LIMIT 0" in query else expected.to_arrow()
        sink = io.BytesIO()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=2)
        payload = sink.getvalue()
        headers = {}
        if content_encoding == "gzip":
            payload = gzip.compress(payload)
        elif content_encoding == "deflate":
            payload = zlib.compress(payload)
        if content_encoding is not None:
            headers["Content-Encoding"] = content_encoding

        response = requests.Response()
        response.status_code = 200
        response.headers.update(headers)
        # Match Requests' streaming transport: raw reads do not decode by default.
        response.raw = HTTPResponse(body=io.BytesIO(payload), headers=headers, preload_content=False, decode_content=False)
        responses.append(response)
        return response

    monkeypatch.setattr(ch.requests, "post", post)
    try:
        result = ch.scan_clickhouse("SELECT id, label FROM events", "http://clickhouse", {}).collect()
        assert_frame_equal(result, expected)
        assert len(queries) == 2
        assert "LIMIT 0" in queries[0]
        assert all(query.endswith("FORMAT ArrowStream") for query in queries)
    finally:
        for response in responses:
            response.close()
