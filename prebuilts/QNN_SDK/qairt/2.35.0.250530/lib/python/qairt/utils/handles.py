# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import weakref
from typing import Any, Callable, Dict, Optional


class Handle:
    """
    A handle that enables easy removal and tracking of objects

    Attributes:
        handles_ref: A weak reference to the dictionary of all handles to handle references
        id: The unique id of the handle
        next_id: The next id to be assigned to a handle
    """

    next_id: int = 0

    def __init__(self, handles: Optional[Dict[int, Any]] = None, handle_ref: Any = None):
        if handles is None:
            handles = {}
        self.handles_ref: weakref.ReferenceType[Dict[int, Any]] = weakref.ref(handles)
        self.id = Handle.next_id
        if handle_ref is not None:
            handles = self.handles_ref()
            if handles is not None:
                handles[self.id] = handle_ref
        Handle.next_id += 1

    def abandon(self) -> None:
        """
        Removes a reference to a handle if it is used in a mapping.
        """
        handles = self.handles_ref()
        if handles is not None and self.id in handles:
            del handles[self.id]


class HookHandle(Handle):
    """A handle that enables easy removal and tracking of hooks"""

    def __init__(self, hook_handles: Dict[int, Any], hook: Callable):
        """
        Initializes a new HookHandle object. A handle holds a reference
        to a mapping of handle to hook, such that a hook may be removed
        from the mapping using handle.abandon().
        """
        super().__init__(hook_handles, hook)
