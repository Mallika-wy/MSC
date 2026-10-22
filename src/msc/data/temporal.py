"""Training queries use the release's affine timeline and token semantics."""
import numpy as np
from data_pipeline.schema.time_map import TimeMap


def select_tokens(times, link, start, end, context=0., semantics='point'):
    mapping = TimeMap(link.time_scale, link.time_offset_sec)
    times = np.asarray(times)
    query_start = max(start-context,getattr(link,'motion_valid_start_sec',float('-inf')))
    query_end = min(end+context,getattr(link,'motion_valid_end_sec',float('inf')))
    lo,hi = mapping.to_asset([query_start,query_end])
    if semantics == 'point':
        first,last = np.searchsorted(times,[lo,hi],side='left')
    else:
        # Intervals may overlap. Search a prefix maximum of ends safely.
        first = np.searchsorted(np.maximum.accumulate(times[:,1]),lo,side='right')
        last = np.searchsorted(times[:,0],hi,side='left')
    subset = times[first:last]
    selected = mapping.select(subset,query_start,query_end,semantics=semantics)
    indices = np.arange(first,last)[selected]
    local = mapping.to_motion(times[indices])-start
    return indices,local


def covers(times, start, end, semantics, rate=None):
    """Require full temporal support; point support uses measured spacing."""
    times = np.asarray(times)
    if len(times) == 0:
        return False
    if semantics == 'interval':
        cursor = start
        for lo, hi in times:
            if lo > cursor + 1e-6:
                return False
            cursor = max(cursor, hi)
        return cursor >= end - 1e-6
    spacing = np.median(np.diff(times)) if len(times) > 1 else (1 / rate if rate else 0)
    return bool(times[0] - spacing / 2 <= start + 1e-6 and
                times[-1] + spacing / 2 >= end - 1e-6 and
                (len(times) < 2 or np.max(np.diff(times)) <= 2 * spacing + 1e-6))
