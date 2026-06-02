from typing import Iterable, Optional

from tqdm.auto import tqdm

# Visual configuration ---------------------------------------------------
INSTACART_GREEN = "#05ad46"
BAR_WIDTH_CHARS = 60
TERMINAL_WIDTH = 200
# tqdm internally treats "%" as a format specifier; "%%" prints as "%".
# The user-facing spec already contains "%%" so we keep that mapping.
BAR_FORMAT = (
    "{desc:30}: {percentage:3.0f}%%|{bar:60}| "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)


def instacart_tqdm(
    iterable: Optional[Iterable] = None,
    *,
    desc: str = "",
    total: Optional[int] = None,
    **kwargs,
) -> tqdm:
    """Return a tqdm progress bar with the project-wide visual identity.

    All other ``tqdm`` keyword arguments are forwarded untouched so call
    sites can opt into ``disable=``, ``mininterval=``, etc. as needed.
    """
    kwargs.setdefault("colour", INSTACART_GREEN)
    kwargs.setdefault("ncols", TERMINAL_WIDTH)
    kwargs.setdefault("bar_format", BAR_FORMAT)
    if "total" not in kwargs and total is not None:
        kwargs["total"] = total
    return tqdm(iterable, desc=desc, **kwargs)

