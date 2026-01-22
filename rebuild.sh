#!/bin/bash
# Clean rebuild script for PTM with yaml-cpp support

set -e  # Exit on error

echo "=== PTM3D Clean Rebuild with YAML Config Support ==="
echo ""

# Check if we're in project root
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Must be run from project root directory"
    exit 1
fi

# Detect yaml-cpp
echo "1. Detecting yaml-cpp installation..."
YAML_FOUND=false
YAML_INCLUDE=""
YAML_LIB=""

# Check Homebrew
if command -v brew &> /dev/null; then
    BREW_PREFIX=$(brew --prefix 2>/dev/null || echo "")
    if [ -n "$BREW_PREFIX" ]; then
        if [ -f "$BREW_PREFIX/include/yaml-cpp/yaml.h" ]; then
            YAML_INCLUDE="$BREW_PREFIX/include"
            # Find library
            if [ -f "$BREW_PREFIX/lib/libyaml-cpp.dylib" ]; then
                YAML_LIB="$BREW_PREFIX/lib/libyaml-cpp.dylib"
                YAML_FOUND=true
                echo "   ✓ Found yaml-cpp (Homebrew)"
            elif [ -f "$BREW_PREFIX/lib/libyaml-cpp.a" ]; then
                YAML_LIB="$BREW_PREFIX/lib/libyaml-cpp.a"
                YAML_FOUND=true
                echo "   ✓ Found yaml-cpp (Homebrew - static)"
            fi
        fi
    fi
fi

# Check conda if Homebrew didn't work
if [ "$YAML_FOUND" = false ] && [ -n "$CONDA_PREFIX" ]; then
    if [ -f "$CONDA_PREFIX/include/yaml-cpp/yaml.h" ]; then
        YAML_INCLUDE="$CONDA_PREFIX/include"
        if [ -f "$CONDA_PREFIX/lib/libyaml-cpp.dylib" ]; then
            YAML_LIB="$CONDA_PREFIX/lib/libyaml-cpp.dylib"
            YAML_FOUND=true
            echo "   ✓ Found yaml-cpp (Conda)"
        elif [ -f "$CONDA_PREFIX/lib/libyaml-cpp.a" ]; then
            YAML_LIB="$CONDA_PREFIX/lib/libyaml-cpp.a"
            YAML_FOUND=true
            echo "   ✓ Found yaml-cpp (Conda - static)"
        fi
    fi
fi

if [ "$YAML_FOUND" = false ]; then
    echo "   ✗ yaml-cpp not found!"
    echo ""
    echo "   To install:"
    echo "     Homebrew: brew install yaml-cpp"
    echo "     Conda:    conda install -c conda-forge yaml-cpp"
    echo ""
    read -p "   Continue without yaml-cpp? (config support will be disabled) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "     Include: $YAML_INCLUDE"
    echo "     Library: $YAML_LIB"
fi

# Clean build directory
echo ""
echo "2. Cleaning build directory..."
if [ -d "build" ]; then
    rm -rf build/*
    echo "   ✓ Build directory cleaned"
else
    mkdir -p build
    echo "   ✓ Build directory created"
fi

# Configure
echo ""
echo "3. Configuring CMake..."
cd build

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"

if [ "$YAML_FOUND" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DYAML_CPP_INCLUDE_DIR=$YAML_INCLUDE"
    CMAKE_ARGS="$CMAKE_ARGS -DYAML_CPP_LIBRARY=$YAML_LIB"
fi

echo "   Running: cmake .. $CMAKE_ARGS"
cmake .. $CMAKE_ARGS

if [ $? -ne 0 ]; then
    echo "   ✗ CMake configuration failed"
    exit 1
fi
echo "   ✓ CMake configuration successful"

# Build
echo ""
echo "4. Building..."
NCORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "   Using $NCORES cores"

make -j$NCORES 2>&1 | tee build.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "   ✗ Build failed. Check build.log for details"
    exit 1
fi

echo ""
echo "   ✓ Build successful"

# Test
echo ""
echo "5. Testing executable..."
if [ -f "bin/ptm_thermal_sim" ]; then
    echo "   ✓ Executable found: bin/ptm_thermal_sim"
    
    # Check if yaml-cpp is linked
    if otool -L bin/ptm_thermal_sim | grep -q yaml-cpp; then
        echo "   ✓ yaml-cpp is linked"
    else
        echo "   ℹ yaml-cpp not linked (config support disabled)"
    fi
    
    # Test help
    if ./bin/ptm_thermal_sim --help &> /dev/null; then
        echo "   ✓ Help command works"
    else
        echo "   ⚠ Help command failed"
    fi
    
    # Test template generation if yaml-cpp available
    if [ "$YAML_FOUND" = true ]; then
        if ./bin/ptm_thermal_sim --generate-template /tmp/test_template.yaml &> /dev/null; then
            echo "   ✓ Template generation works"
            rm -f /tmp/test_template.yaml
        else
            echo "   ⚠ Template generation failed (yaml-cpp may not be properly linked)"
        fi
    fi
else
    echo "   ✗ Executable not found"
    exit 1
fi

echo ""
echo "=== Build Complete ==="
echo ""
echo "Executable: $(pwd)/bin/ptm_thermal_sim"
if [ "$YAML_FOUND" = true ]; then
    echo "YAML config support: ENABLED"
    echo ""
    echo "Try:"
    echo "  ./bin/ptm_thermal_sim --generate-template"
    echo "  ./bin/ptm_thermal_sim --config myconfig.yaml"
else
    echo "YAML config support: DISABLED"
    echo ""
    echo "All command-line arguments still work normally."
fi
echo ""
