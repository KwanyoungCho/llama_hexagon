//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_PREPROC_HPP
#define QUALLA_DETAIL_PREPROC_HPP

#include <cstdio>
#define QUALLA_ASSERT(x)                                                     \
  do {                                                                       \
    if (!(x)) {                                                              \
      fprintf(stderr, "QUALLA_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
      abort();                                                               \
    }                                                                        \
  } while (0)

#endif  // QUALLA_DETAIL_PREPROC_HPP
