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
APP_MODULES := Genie
APP_CPPFLAGS += -std=c++2a -O3 -Wall -frtti -fexceptions -fvisibility=hidden -DGENIE_API="__attribute__((visibility(\"default\")))" -DSPILLFILL -DQUALLA_ENGINE_QNN_HTP=TRUE -DQUALLA_ENGINE_QNN_CPU=TRUE -DQUALLA_ENGINE_QNN_GPU=TRUE -DFMT_HEADER_ONLY -DGENIE_SAMPLE -DQUALLA_INTERNAL_QNN_SDK
APP_LDFLAGS  += -lc -lm -ldl -Wl,--version-script=GenieSymbols.default -Wl,--strip-all
