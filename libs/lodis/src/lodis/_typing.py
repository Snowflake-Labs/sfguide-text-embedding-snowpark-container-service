from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar

import numpy as np
from numpy._typing import _GenericAlias

T_Item = TypeVar("T_Item")
T_ItemDtype = TypeVar("T_ItemDtype", bound=np.dtype)
T_Key = TypeVar("T_Key", bound=np.dtype)
T_Value = TypeVar("T_Value", bound=np.dtype)
T_KVDtype = TypeVar("T_KVDtype", bound=np.dtype)

# Create alternative to NDArray that allows nonscalar dtypes.
NonscalarType = TypeVar("NonscalarType", bound=np.dtype, covariant=True)
if TYPE_CHECKING:
    NonscalarNDArray = np.ndarray[Any, NonscalarType]
else:
    NonscalarNDArray = _GenericAlias(
        np.ndarray, (Any, _GenericAlias(np.dtype, (NonscalarType,)))
    )
