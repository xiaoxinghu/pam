// What qwen3-asr.cpp's forced aligner takes from ggml's internal header,
// which pam-llama doesn't put on the include path: the log macros, through
// ggml's logger (which llama_log_set sets).
#pragma once
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif
GGML_API void ggml_log_internal(enum ggml_log_level level, const char * format, ...);
#ifdef __cplusplus
}
#endif

#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
