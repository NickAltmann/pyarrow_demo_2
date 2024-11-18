from typing import Sequence

from .pysect import *


__doc__ = pysect.__doc__
if hasattr(pysect, "__all__"):
    __all__ = pysect.__all__


def bisect_python(sorted_list: Sequence[float], value: int, low: int=0, high: int=None):
    if high is None:
        high = len(sorted_list) - 1
    
    if low > high:
        return low
    
    mid = (low + high) // 2
    
    if sorted_list[mid] < value:
        return bisect_python(sorted_list, value, mid + 1, high)
    else:
        return bisect_python(sorted_list, value, low, mid - 1)
