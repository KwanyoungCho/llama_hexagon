#=============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

APP_ABI      := arm64-v8a
APP_STL      := c++_shared
APP_PLATFORM := android-21
APP_MODULES := genie-t2t-run
APP_CPPFLAGS += -std=c++2a -O3 -Wall -frtti -fexceptions -fvisibility=hidden 
APP_LDFLAGS  += -lc -lm -ldl
