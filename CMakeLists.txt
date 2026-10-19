cmake_minimum_required(VERSION 3.22)

# Must be set before project() to take effect.
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")

project(SIDVae VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#-------------------------------------------------------------------------------
# Options
#-------------------------------------------------------------------------------
option(SIDVAE_WITH_ONNX "Run the VAE decoder with ONNX Runtime (OFF = fallback crossfade only)" ON)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "Unpacked ONNX Runtime release (contains include/ and lib/)")
set(SIDVAE_JUCE_TAG "9.0.4" CACHE STRING "JUCE git tag fetched when there is no JUCE/ folder")
option(SIDVAE_INSTALL_AFTER_BUILD "macOS: copy VST3/AU into ~/Library/Audio/Plug-Ins after building" ON)

#-------------------------------------------------------------------------------
# JUCE: use ./JUCE (e.g. a git submodule) if present, otherwise fetch it.
#-------------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/JUCE/CMakeLists.txt")
    add_subdirectory(JUCE)
else()
    include(FetchContent)
    FetchContent_Declare(JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        ${SIDVAE_JUCE_TAG}
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(JUCE)
endif()

#-------------------------------------------------------------------------------
# Plugin
#-------------------------------------------------------------------------------
juce_add_plugin(SIDVae
    COMPANY_NAME                "hayakzan"
    BUNDLE_ID                   com.hayakzan.sidvae
    PLUGIN_MANUFACTURER_CODE    Hyak
    PLUGIN_CODE                 Sdva
    IS_SYNTH                    TRUE
    NEEDS_MIDI_INPUT            TRUE      # hosts expect instruments to take MIDI; unused for now
    NEEDS_MIDI_OUTPUT           FALSE
    IS_MIDI_EFFECT              FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD     FALSE     # done below, after the model + runtime are bundled
    FORMATS                     AU VST3 Standalone
    PRODUCT_NAME                "SIDVae")

target_sources(SIDVae PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/core/SidVaeDecoder.cpp)

target_include_directories(SIDVae PRIVATE Source)

target_compile_definitions(SIDVae PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    SIDVAE_DEV_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Models")

target_link_libraries(SIDVae
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_gui_basics
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

#-------------------------------------------------------------------------------
# ONNX Runtime
#-------------------------------------------------------------------------------
set(ORT_RUNTIME_FILES "")
if(SIDVAE_WITH_ONNX)
    find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime"
        NO_DEFAULT_PATH)
    find_library(ORT_LIBRARY onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)

    if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime not found in ${ONNXRUNTIME_ROOT}\n"
            "Download onnxruntime-osx-universal2-<version>.tgz from\n"
            "  https://github.com/microsoft/onnxruntime/releases\n"
            "and unpack it so that third_party/onnxruntime/include and /lib exist,\n"
            "or pass -DONNXRUNTIME_ROOT=/path/to/it, or build without the VAE: -DSIDVAE_WITH_ONNX=OFF")
    endif()

    target_include_directories(SIDVae PRIVATE "${ORT_INCLUDE_DIR}")
    target_link_libraries(SIDVae PRIVATE "${ORT_LIBRARY}")
    target_compile_definitions(SIDVae PRIVATE SIDVAE_WITH_ONNX=1)

    if(APPLE)
        # The bundled copy must carry the dylib's install name (e.g. @rpath/libonnxruntime.1.dylib
        # in 1.31, @rpath/libonnxruntime.1.17.0.dylib in older releases), so ask otool for it.
        get_filename_component(ORT_DYLIB_NAME "${ORT_LIBRARY}" NAME)
        execute_process(COMMAND otool -D "${ORT_LIBRARY}"
                        OUTPUT_VARIABLE _ort_id OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_ort_id MATCHES "@rpath/([^\n]+)$")
            set(ORT_DYLIB_NAME "${CMAKE_MATCH_1}")
        endif()
        message(STATUS "SIDVae: bundling ONNX Runtime as Frameworks/${ORT_DYLIB_NAME}")
    elseif(WIN32)
        file(GLOB ORT_RUNTIME_FILES "${ONNXRUNTIME_ROOT}/lib/onnxruntime*.dll")
    else()
        file(GLOB ORT_RUNTIME_FILES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*")
    endif()
    message(STATUS "SIDVae: ONNX Runtime at ${ONNXRUNTIME_ROOT}")
else()
    target_compile_definitions(SIDVae PRIVATE SIDVAE_WITH_ONNX=0)
    message(STATUS "SIDVae: building WITHOUT ONNX Runtime (fallback crossfade only)")
endif()

#-------------------------------------------------------------------------------
# Bundle the model, its metadata and the ONNX Runtime library into each format.
#-------------------------------------------------------------------------------
set(SIDVAE_MODEL_FILES "")
file(GLOB SIDVAE_MODEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Models/*")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sid_vae_meta.json")
    list(APPEND SIDVAE_MODEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/sid_vae_meta.json")
endif()

foreach(fmt VST3 AU Standalone)
    set(tgt SIDVae_${fmt})
    if(NOT TARGET ${tgt})
        continue()
    endif()

    if(APPLE)
        set(contents "$<TARGET_BUNDLE_CONTENT_DIR:${tgt}>")
        set_target_properties(${tgt} PROPERTIES
            BUILD_RPATH   "@loader_path/../Frameworks"
            INSTALL_RPATH "@loader_path/../Frameworks")

        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${contents}/Resources/Models"
            VERBATIM)
        if(SIDVAE_MODEL_FILES)
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SIDVAE_MODEL_FILES} "${contents}/Resources/Models/"
                VERBATIM)
        endif()
        if(SIDVAE_WITH_ONNX)
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${contents}/Frameworks"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_LIBRARY}" "${contents}/Frameworks/${ORT_DYLIB_NAME}"
                VERBATIM)
        endif()
        # Re-sign (ad hoc) after changing the bundle, or Apple Silicon hosts may refuse to load it.
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND codesign --force --deep --sign - "$<TARGET_BUNDLE_DIR:${tgt}>"
            VERBATIM)

        if(SIDVAE_INSTALL_AFTER_BUILD AND (fmt STREQUAL "VST3" OR fmt STREQUAL "AU"))
            if(fmt STREQUAL "VST3")
                set(dest "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/SIDVae.vst3")
            else()
                set(dest "$ENV{HOME}/Library/Audio/Plug-Ins/Components/SIDVae.component")
            endif()
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E rm -rf "${dest}"
                COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_BUNDLE_DIR:${tgt}>" "${dest}"
                VERBATIM)
        endif()
    else()
        # Windows / Linux: untested. Model and runtime go next to the plugin binary.
        set(bindir "$<TARGET_FILE_DIR:${tgt}>")
        if(UNIX)
            set_target_properties(${tgt} PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
        endif()
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${bindir}/Models"
            VERBATIM)
        if(SIDVAE_MODEL_FILES)
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SIDVAE_MODEL_FILES} "${bindir}/Models/"
                VERBATIM)
        endif()
        if(ORT_RUNTIME_FILES)
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_RUNTIME_FILES} "${bindir}/"
                VERBATIM)
        endif()
    endif()
endforeach()
