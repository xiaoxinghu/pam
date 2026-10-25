# Included right after NeMo-Speech's `project()` (CMAKE_PROJECT_nemo_speech_INCLUDE,
# from build.rs). Its src/asr/CMakeLists.txt makes the two ASR libraries SHARED;
# they are built STATIC instead, so that pam links them into its binaries.
function(add_library name)
    if(name MATCHES "^nemo_speech_asr(_c)?$" AND "${ARGV1}" STREQUAL "SHARED")
        list(SUBLIST ARGN 1 -1 sources)
        _add_library(${name} STATIC ${sources})
    else()
        _add_library(${ARGV})
    endif()
endfunction()
