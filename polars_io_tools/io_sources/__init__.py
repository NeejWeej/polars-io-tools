import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

import polars as pl

from .base import *
from .concat_named import *
from .join import *
from .lazy_cache import cache, cache as _lazy_cache
from .lazy_cache_memory import *
from .lazy_cache_parquet import *
from .lazy_datadog_reader import *
from .lazy_debug import debug, debug as _lazy_debug
from .lazy_iter_rows import *
from .lazy_probe import probe, probe as _lazy_probe
from .partitions import *
from .pushdown_combine import *
from .pushdown_pivot import *
from .pushdown_unpivot import *
from .translated_source import *
from .ts import *
from .util import *

# These entry points are backed by heavy optional dependencies, so their modules
# are not imported at package load time. Each is a thin delegator that imports its
# module on first call; the TYPE_CHECKING imports give type checkers real signatures.
if TYPE_CHECKING:
    from .delta_io import scan_delta, sink_delta
    from .lazy_clickhouse_reader import scan_clickhouse
    from .lazy_clickhouse_writer import sink_clickhouse
    from .lazy_data_generator import scan_synthetic_panel, scan_synthetic_regression
    from .lazy_narwhals_reader import from_narwhals
    from .lazy_sql_reader import scan_db
else:

    def scan_db(*args, **kwargs):
        from .lazy_sql_reader import scan_db as _f

        return _f(*args, **kwargs)

    def scan_clickhouse(*args, **kwargs):
        from .lazy_clickhouse_reader import scan_clickhouse as _f

        return _f(*args, **kwargs)

    def sink_clickhouse(*args, **kwargs):
        from .lazy_clickhouse_writer import sink_clickhouse as _f

        return _f(*args, **kwargs)

    def scan_synthetic_panel(*args, **kwargs):
        from .lazy_data_generator import scan_synthetic_panel as _f

        return _f(*args, **kwargs)

    def scan_synthetic_regression(*args, **kwargs):
        from .lazy_data_generator import scan_synthetic_regression as _f

        return _f(*args, **kwargs)

    def from_narwhals(*args, **kwargs):
        from .lazy_narwhals_reader import from_narwhals as _f

        return _f(*args, **kwargs)

    def scan_delta(*args, **kwargs):
        from .delta_io import scan_delta as _f

        return _f(*args, **kwargs)

    def sink_delta(*args, **kwargs):
        from .delta_io import sink_delta as _f

        return _f(*args, **kwargs)


if TYPE_CHECKING:
    # Resolve the @functools.wraps(...) targets on the PIOTOperations methods for static
    # analysis. Most targets are provided at runtime by the `from .X import *` star imports
    # above and are re-declared here so type checkers can see them (TC004 suppressed). The
    # three lazily-imported targets (execute_on_ray, sink_clickhouse, sink_delta) resolve to
    # the real functions here but to no-op stubs at runtime, so wraps can run at class-
    # definition time without importing their optional dependencies.
    from .delta_io import sink_delta as _sink_delta_proto
    from .join import filtered_join, filtered_join_asof  # noqa: TC004
    from .lazy_clickhouse_writer import sink_clickhouse as _sink_clickhouse_proto
    from .lazy_iter_rows import iter_rows  # noqa: TC004
    from .lazy_ray import execute_on_ray as _execute_on_ray_proto
    from .partitions import KeyPartitions, ReadPartition, by_key, by_range, by_time, by_value
    from .pushdown_combine import FilterSpec, pushdown_combine
    from .pushdown_pivot import pushdown_pivot
    from .pushdown_unpivot import pushdown_unpivot
    from .ts import ts_with_columns  # noqa: TC004
    from .util import filter_no_pushdown, with_columns_topo  # noqa: TC004
else:

    def _execute_on_ray_proto(*_a, **_kw): ...

    def _sink_clickhouse_proto(*_a, **_kw): ...

    def _sink_delta_proto(*_a, **_kw): ...


@pl.api.register_lazyframe_namespace("piot")
class PIOTOperations:
    # If this flag is true, will replace optimized operations with their standard polars equivalents so that
    # explain can be run on the dataframe. Note that these modified explain plans will not reflect the additional
    # optimizations we are performing, but should return the same answers.
    _DISABLE_OPTIMIZATIONS: bool = False

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    @functools.wraps(_lazy_debug)
    def debug(self, *args, **kwargs) -> pl.LazyFrame:
        return _lazy_debug(self._lf, *args, **kwargs)

    @functools.wraps(_lazy_probe)
    def probe(self, *args, **kwargs) -> pl.LazyFrame:
        return _lazy_probe(self._lf, *args, **kwargs)

    @functools.wraps(_lazy_cache)
    def cache(self, *args, **kwargs) -> pl.LazyFrame:
        if self._DISABLE_OPTIMIZATIONS:
            return self._lf
        return _lazy_cache(self._lf, *args, **kwargs)

    @functools.wraps(filtered_join)
    def filtered_join(self, *args, **kwargs) -> pl.LazyFrame:
        if self._DISABLE_OPTIMIZATIONS:
            return self._lf.join(*args, **kwargs)
        return filtered_join(self._lf, *args, **kwargs)

    @functools.wraps(cache_parquet)
    def cache_parquet(self, *args, **kwargs) -> pl.LazyFrame:
        return cache_parquet(self._lf, *args, **kwargs)

    @functools.wraps(cache_memory)
    def cache_memory(self, *args, **kwargs) -> pl.LazyFrame:
        return cache_memory(self._lf, *args, **kwargs)

    @functools.wraps(_execute_on_ray_proto)
    def execute_on_ray(self, *args, **kwargs) -> pl.LazyFrame:
        # heavy import happens only when the user calls the method
        from .lazy_ray import execute_on_ray

        return execute_on_ray(self._lf, *args, **kwargs)

    @functools.wraps(filtered_join_asof)
    def filtered_join_asof(self, *args, **kwargs) -> pl.LazyFrame:
        if self._DISABLE_OPTIMIZATIONS:
            return self._lf.join_asof(*args, **kwargs)
        return filtered_join_asof(self._lf, *args, **kwargs)

    @functools.wraps(ts_with_columns)
    def ts_with_columns(self, *args, **kwargs) -> pl.LazyFrame:
        if self._DISABLE_OPTIMIZATIONS:
            kwargs["_disable_optimizations"] = True
        return ts_with_columns(self._lf, *args, **kwargs)

    @functools.wraps(filter_no_pushdown)
    def filter_no_pushdown(self, *args, **kwargs) -> pl.LazyFrame:
        if self._DISABLE_OPTIMIZATIONS:
            kwargs["_disable_optimizations"] = True
        return filter_no_pushdown(self._lf, *args, **kwargs)

    @functools.wraps(with_columns_topo)
    def with_columns_topo(self, *args, **kwargs) -> pl.LazyFrame:
        return with_columns_topo(self._lf, *args, **kwargs)

    @functools.wraps(_sink_delta_proto)
    def sink_delta(self, *args, **kwargs):
        # heavy import happens only when the user calls the method
        from .delta_io import sink_delta

        return sink_delta(self._lf, *args, **kwargs)

    @functools.wraps(_sink_clickhouse_proto)
    def sink_clickhouse(self, *args, **kwargs):
        # heavy import happens only when the user calls the method
        from .lazy_clickhouse_writer import sink_clickhouse

        return sink_clickhouse(self._lf, *args, **kwargs)

    @functools.wraps(iter_rows)
    def iter_rows(self, *args, **kwargs):
        return iter_rows(self._lf, *args, **kwargs)


@contextmanager
def disable_optimizations():
    """Context manager for disabling optimizations in operations.

    This is useful to get a proper polars "explain" plan for a LazyFrame that would otherwise be obscured
    because polars will list all operations as "PYTHON_SCAN".

    It can also be used to test that the results with and without optimizations are the same.
    """
    try:
        PIOTOperations._DISABLE_OPTIMIZATIONS = True
        yield
    finally:
        PIOTOperations._DISABLE_OPTIMIZATIONS = False
