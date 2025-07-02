#=============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

RUST_TARGET := aarch64-linux-android
RUST_SOURCE_DIR := ./src/qualla/tokenizers/rust
# specify compiler
export CXX := clang++-14
export PATH := $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin:$(PATH)
.PHONY: all x86 android clean clean_x86 clean_android
.DEFAULT: x86

all: x86 android

x86: build_x86_tokenizer
	@echo "-------------------- Building genie for x86 -------------------- "
	@$(MAKE) -f make/Makefile.linux-x86_64 CPATH="/usr/include/x86_64-linux-gnu" || (echo "-------------------- genie x86 build failed --------------------"; exit 1; )
	@echo "-------------------- genie x86 build succeeded -------------------- "

android: check_ndk build_android_tokenizer
	@echo "-------------------- Building genie for android -------------------- "
	@$(ANDROID_NDK_ROOT)/ndk-build APP_ALLOW_MISSING_DEPS=true APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=make/Application.mk APP_BUILD_SCRIPT=make/Android.mk || (echo "-------------------- genie android build failed --------------------"; exit 1; )
	@$(rename_target_dirs)
	@echo "-------------------- genie android build succeeded -------------------- "

clean: clean_x86 clean_android

clean_x86:
	@$(MAKE) -f make/Makefile.linux-x86_64 clean

clean_android:
	if [ -d "lib/aarch64-android" ]; then rm -rf lib/aarch64-android; fi
	if [ -d "obj/local" ]; then rm -rf obj/local; fi

# utilities
rename_target_dirs = \
   				 @if [ -d ./lib/aarch64-android ]; then rm -rf ./lib/aarch64-android; fi; \
					 find ./obj/local -type d -execdir rename 's/arm64-v8a/aarch64-android/' '{}' \+ \
					 && mkdir -p lib \
					 && mv ./obj/local/aarch64-android lib/ \
					 && mv ./libs/arm64-v8a/libc++_shared.so lib/aarch64-android/ \
					 && rm -rf ./libs \

check_ndk:
ifeq ($(ANDROID_NDK_ROOT),)
	$(error ERROR: ANDROID_NDK_ROOT not set, skipping compilation for Android platform(s).)
endif

build_x86_tokenizer: $(RUST_SOURCE_DIR)/Cargo.toml
	cargo build --release --manifest-path=$<

build_android_tokenizer: $(RUST_SOURCE_DIR)/Cargo.toml
	cargo build --release --manifest-path=$< --target=$(RUST_TARGET)
