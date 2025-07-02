#=============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

# specify compiler
export CXX := clang++-14
export PATH := $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin:$(PATH)
.PHONY: all x86 android clean clean_x86 clean_android
.DEFAULT: x86

all: x86 android
x86:
	@$(MAKE) -f make/Makefile.linux-x86_64 CPATH="/usr/include/x86_64-linux-gnu" || (echo "-------------------- Failed to build genie-t2t-run for x86 --------------------"; exit 1; )
	@echo "-------------------- genie-t2t-run build for x86 succeeded -------------------- "

android: check_ndk
	@$(ANDROID_NDK_ROOT)/ndk-build APP_ALLOW_MISSING_DEPS=true APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=make/Application.mk APP_BUILD_SCRIPT=make/Android.mk || (echo "-------------------- Failed to build genie-t2t-run for android --------------------"; exit 1; )
	@$(rename_target_dirs)
	@echo "-------------------- genie-t2t-run build for android succeeded-------------------- "

clean: clean_x86 clean_android

clean_x86:
	@$(MAKE) -f make/Makefile.linux-x86_64 clean

clean_android:
	if [ -d "bin/aarch64-android" ]; then rm -rf bin/aarch64-android; fi
	if [ -d "obj/local" ]; then rm -rf obj/local; fi

rename_target_dirs = \
				@if [ -d ./bin/aarch64-android ]; then rm -rf ./bin/aarch64-android; fi; \
				mkdir -p bin/aarch64-android \
				&& mv ./libs/arm64-v8a/genie-t2t-run bin/aarch64-android \
				&& mv ./libs/arm64-v8a/libc++_shared.so bin/aarch64-android \
				&& rm -rf ./libs \


check_ndk:
ifeq ($(ANDROID_NDK_ROOT),)
	$(error ERROR: ANDROID_NDK_ROOT not set, skipping compilation for Android platform(s).)
endif
