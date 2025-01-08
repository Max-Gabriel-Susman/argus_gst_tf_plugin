cmake_minimum_required(VERSION 3.10)
project(gst_tf_detect)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)

# You’ll need to point these to your TensorFlow library location
# Example: set(TensorFlow_INCLUDE /usr/local/include)
# Example: set(TensorFlow_LIB /usr/local/lib/libtensorflow_cc.so)

include_directories(
    ${GST_INCLUDE_DIRS}
    ${TensorFlow_INCLUDE}
)

add_definitions(${GST_CFLAGS_OTHER})

add_library(gsttfdetect SHARED gsttfdetect.cpp)
target_link_libraries(
    gsttfdetect
    ${GST_LIBRARIES}
    ${TensorFlow_LIB}
)

install(TARGETS gsttfdetect LIBRARY DESTINATION lib/gstreamer-1.0)
