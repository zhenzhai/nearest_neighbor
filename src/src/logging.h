/* 
 * File             : logging.h
 * Date             : 2014-5-29
 * Summary          : Provides the logging macros used throughout.
 */
#ifndef LOGGING_
#define LOGGING_

#include <stdio.h>

#define ERROR       (400)
#define WARNING     (300)
#define INFO        (200)
#define FINE        (100)

#define LOG_ERROR(...) 
#define LOG_WARNING(...) 
#define LOG_INFO(...) 
#define LOG_FINE(...) 

#ifdef DEBUG

#if ERROR >= DEBUG
#undef  LOG_ERROR
#define LOG_ERROR(...) (fprintf(stderr, "[ERROR] ") && \
                        fprintf(stderr, __VA_ARGS__))
#endif

#if WARNING >= DEBUG
#undef  LOG_WARNING
#define LOG_WARNING(...) (fprintf(stderr, "[WARNING] ") && \
                          fprintf(stderr, __VA_ARGS__))
#endif

#if INFO >= DEBUG
#undef  LOG_INFO
#define LOG_INFO(...) (fprintf(stderr, "[INFO] ") && \
                       fprintf(stderr, __VA_ARGS__))
#endif

#if FINE >= DEBUG
#undef  LOG_FINE
#define LOG_FINE(...) (fprintf(stderr, "[FINE] ") && \
                       fprintf(stderr, __VA_ARGS__))
#endif

#endif

#endif
