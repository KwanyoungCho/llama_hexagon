#=============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

LOCAL_PATH := $(call my-dir)
SUPPORTED_TARGET_ABI := arm64-v8a x86 x86_64

#============================ Verify Target Info and Application Variables =========================================
ifneq ($(filter $(TARGET_ARCH_ABI),$(SUPPORTED_TARGET_ABI)),)
    ifneq ($(APP_STL), c++_shared)
        $(error Unsupported APP_STL: "$(APP_STL)")
    endif
else
    $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

#============================ Define Common Variables ===============================================================
# PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../../../../../include/QNN
# Include paths
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../include
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../../../../include/Genie
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/include
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../../../../include/QNN
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../../../../include/QNN/HTP
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/tokenizers
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/MmappedFile/include
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-api
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/encoders/text-encoders
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/encoders/image-encoders
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-cpu
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-gpu
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-htp
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/pipeline

#========================== Define T2T Lib variables =============================================
include $(CLEAR_VARS)
LOCAL_MODULE := tokenizers_capi
LOCAL_SRC_FILES := ../src/qualla/tokenizers/rust/target/aarch64-linux-android/release/libtokenizers_capi.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_C_INCLUDES               := $(PACKAGE_C_INCLUDES)
MY_SRC_FILES                   := $(wildcard $(LOCAL_PATH)/../src/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/pipeline/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/dialogs/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/encoders/text-encoders/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/encoders/image-encoders/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/MmappedFile/src/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-api/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-cpu/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-gpu/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-htp/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/utils/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/loggers/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/samplers/*.cpp)

LOCAL_MODULE                   := libGenie
LOCAL_SRC_FILES                := $(subst make/,,$(MY_SRC_FILES))
LOCAL_STATIC_LIBRARIES         := tokenizers_capi
LOCAL_LDLIBS                   := -llog
include $(BUILD_SHARED_LIBRARY)
