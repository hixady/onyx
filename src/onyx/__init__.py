from onyx.core.registry import register, lookup, list_types, scan_entry_points
from onyx.core.runner import run

scan_entry_points()

import onyx.models.rvc
import onyx.models.mdx

__all__ = ["register", "lookup", "list_types", "run"]
