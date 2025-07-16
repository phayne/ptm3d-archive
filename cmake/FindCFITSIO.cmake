# - Find CFITSIO
# Find the CFITSIO includes and library
# This module defines
#  CFITSIO_INCLUDE_DIR, where to find fitsio.h
#  CFITSIO_LIBRARIES, the libraries to link against
#  CFITSIO_FOUND, if false, do not try to use CFITSIO

# First try to use pkg-config to find cfitsio
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_CFITSIO cfitsio)
  set(CFITSIO_DEFINITIONS ${PC_CFITSIO_CFLAGS_OTHER})
endif()

# Find the include directory
find_path(CFITSIO_INCLUDE_DIR 
    NAMES fitsio.h
    PATHS
    ${PC_CFITSIO_INCLUDEDIR}
    ${PC_CFITSIO_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    /opt/local/include
    $ENV{CONDA_PREFIX}/include
    PATH_SUFFIXES cfitsio
)

# Find the library
find_library(CFITSIO_LIBRARY
    NAMES cfitsio
    PATHS
    ${PC_CFITSIO_LIBDIR}
    ${PC_CFITSIO_LIBRARY_DIRS}
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    $ENV{CONDA_PREFIX}/lib
)

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CFITSIO DEFAULT_MSG CFITSIO_LIBRARY CFITSIO_INCLUDE_DIR)

# Hide internal variables
mark_as_advanced(CFITSIO_INCLUDE_DIR CFITSIO_LIBRARY)

# Set output variables
set(CFITSIO_LIBRARIES ${CFITSIO_LIBRARY})
set(CFITSIO_INCLUDE_DIRS ${CFITSIO_INCLUDE_DIR})