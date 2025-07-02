# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import platform
import sys


if platform.system() == "Linux":
    if platform.machine() == "x86_64":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'linux-x86_64'))
    else:
        raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), platform.machine()))
elif platform.system() == "Windows":
    if "AMD64" in platform.processor() or "Intel64" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-x86_64'))
    elif "ARMv8" in platform.processor():
        os.add_dll_directory(os.path.join(os.environ["QNN_SDK_ROOT"], "lib", "arm64x-windows-msvc"))
        os.environ['PATH'] = os.path.join(os.environ["QNN_SDK_ROOT"], "lib", "arm64x-windows-msvc") + os.pathsep + os.environ['PATH']
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-arm64ec'))
    else:
        cpu_isa = platform.processor().split()[0]
        raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), cpu_isa))
else:
    raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), platform.machine()))


try:
    if sys.version_info[:2] == (3, 6):
        import libPyNetRun36 as py_net_run
    elif sys.version_info[:2] == (3, 8):
        import libPyNetRun38 as py_net_run
    else:
        import libPyNetRun as py_net_run
except ImportError as e:
    try:
        if sys.version_info[:2] == (3, 6):
            from . import libPyNetRun36 as py_net_run
        elif sys.version_info[:2] == (3, 8):
            from . import libPyNetRun38 as py_net_run
        else:
            from . import libPyNetRun as py_net_run
    except ImportError:
        print("Cannot import pybind library: libPyNetRun")
        raise e
