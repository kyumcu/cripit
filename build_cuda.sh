#!/bin/bash
#
# build_cuda.sh - Comprehensive build script for CUDA-enabled pywhispercpp
#
# This script builds and installs pywhispercpp with CUDA support for GPU
# acceleration in CripIt. It includes:
# - Pre-flight system checks
# - CUDA environment validation
# - Backup of existing installation
# - Build with detailed logging
# - Post-build verification
# - Troubleshooting assistance
#
# Usage:
#   bash build_cuda.sh           # Interactive build
#   bash build_cuda.sh --force   # Skip confirmation prompts
#   bash build_cuda.sh --verbose # Extra verbose output
#
# Recommended command set (conda + CUDA_HOME):
#   source "$HOME/miniforge3/etc/profile.d/conda.sh"
#   conda activate py_cripit
#   export CUDA_HOME=/usr/lib/cuda
#   bash ./build_cuda.sh --force --verbose
#
# Verify after build:
#   python -c 'from utils.cuda_utils import CUDAManager; print(CUDAManager().validate_cuda_setup())'
#

set -e  # Exit on error

# If invoked with sh, re-exec with bash (this script uses bash features)
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LOG_FILE="/tmp/pywhispercpp_cuda_build.log"
BACKUP_DIR="/tmp/pywhispercpp_backup_$(date +%Y%m%d_%H%M%S)"
FORCE_MODE=false
VERBOSE_MODE=false
NVCC_PATH=""

# Resolve repo root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force    Skip confirmation prompts"
            echo "  --verbose  Enable extra verbose output"
            echo "  --help     Show this help message"
            echo ""
            echo "Recommended command set (conda + CUDA_HOME):"
            echo "  source \"\$HOME/miniforge3/etc/profile.d/conda.sh\""
            echo "  conda activate py_cripit"
            echo "  export CUDA_HOME=/usr/lib/cuda"
            echo "  bash ./build_cuda.sh --force --verbose"
            echo ""
            echo "Verify after build:"
            echo "  python -c 'from utils.cuda_utils import CUDAManager; print(CUDAManager().validate_cuda_setup())'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
    echo "[STEP] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

# Print header
print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         CripIt CUDA Build Script for pywhispercpp         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    log_info "Build log will be saved to: $LOG_FILE"
    echo ""
}

# Check if running in correct conda environment
check_conda_env() {
    log_step "Checking Conda environment..."
    
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        log_error "Not in a conda environment. Please activate an environment first:"
        log_error "  conda activate py_cripit"
        exit 1
    fi
    
    log_info "Active conda environment: $CONDA_DEFAULT_ENV"
    
    # Check if we're in the right environment
    if [ "$CONDA_DEFAULT_ENV" != "py_cripit" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
        log_warn "Current environment is '$CONDA_DEFAULT_ENV', not 'py_cripit'"
        if [ "$FORCE_MODE" = false ]; then
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
}

# Check Python version
check_python() {
    log_step "Checking Python installation..."
    
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"
    
    # Check if Python version is 3.8+
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        log_error "Python 3.8 or higher is required (found $PYTHON_VERSION)"
        exit 1
    fi
    
    log_info "Python version is compatible"
}

# Check CUDA availability
check_cuda() {
    log_step "Checking CUDA installation..."
    
    # Check CUDA_HOME
    if [ -z "$CUDA_HOME" ]; then
        # Try to auto-detect
        log_warn "CUDA_HOME not set, attempting to auto-detect..."
        
        # Check common locations
        CUDA_CANDIDATES=(
            "/usr/local/cuda"
            "/usr/local/cuda-12.0"
            "/usr/local/cuda-11.8"
            "/usr/local/cuda-11.7"
            "/usr/lib/cuda"
            "/opt/cuda"
        )
        
        for candidate in "${CUDA_CANDIDATES[@]}"; do
            if [ -d "$candidate" ]; then
                export CUDA_HOME="$candidate"
                log_info "Auto-detected CUDA at: $CUDA_HOME"
                break
            fi
        done
    fi
    
    if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
        log_error "CUDA_HOME is not set or directory does not exist"
        log_error "Please install CUDA Toolkit and set CUDA_HOME environment variable"
        log_error "Example: export CUDA_HOME=/usr/local/cuda"
        exit 1
    fi
    
    log_info "CUDA_HOME: $CUDA_HOME"
    
    # Check for nvcc
    if [ -f "$CUDA_HOME/bin/nvcc" ]; then
        NVCC_PATH="$CUDA_HOME/bin/nvcc"
    elif command -v nvcc &> /dev/null; then
        NVCC_PATH=$(command -v nvcc)
    else
        log_error "NVCC compiler not found"
        log_error "Expected at: $CUDA_HOME/bin/nvcc"
        exit 1
    fi
    
    log_info "NVCC found: $NVCC_PATH"

    # If CUDA_HOME/bin/nvcc doesn't exist (common on some distros), prefer actual nvcc
    if [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
        log_warn "CUDA_HOME/bin/nvcc not found; will use nvcc from PATH"
    fi
    
    # Get CUDA version
    CUDA_VERSION=$($NVCC_PATH --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    log_info "CUDA Version: $CUDA_VERSION"
    
    # Check CUDA version compatibility (11.8+ or 12.x recommended)
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    if [ "$CUDA_MAJOR" -lt 11 ]; then
        log_warn "CUDA $CUDA_VERSION detected. CUDA 11.8+ or 12.x is recommended"
        if [ "$FORCE_MODE" = false ]; then
            read -p "Continue with older CUDA version? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    # Check for nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        log_info "nvidia-smi available"
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        log_info "Detected $GPU_COUNT GPU(s)"
    else
        log_warn "nvidia-smi not found in PATH"
    fi
    
    # Verify CUDA libraries exist
    if [ -d "$CUDA_HOME/lib64" ]; then
        log_info "CUDA libraries found at: $CUDA_HOME/lib64"
    elif [ -d "$CUDA_HOME/lib" ]; then
        log_info "CUDA libraries found at: $CUDA_HOME/lib"
    else
        log_error "CUDA libraries not found in $CUDA_HOME/lib64 or $CUDA_HOME/lib"
        exit 1
    fi
}

# Check for required build tools
check_build_tools() {
    log_step "Checking build tools..."
    
    # Check for cmake
    if ! command -v cmake &> /dev/null; then
        log_error "cmake is not installed. Please install it first:"
        log_error "  Ubuntu/Debian: sudo apt-get install cmake"
        log_error "  macOS: brew install cmake"
        exit 1
    fi
    
    CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
    log_info "cmake version: $CMAKE_VERSION"
    
    # Check for git
    if ! command -v git &> /dev/null; then
        log_error "git is not installed. Please install it first"
        exit 1
    fi
    
    log_info "git is available"
    
    # Check for build-essential (gcc/g++)
    if ! command -v gcc &> /dev/null; then
        log_warn "gcc not found. You may need build tools:"
        log_warn "  Ubuntu/Debian: sudo apt-get install build-essential"
        log_warn "  macOS: xcode-select --install"
    else
        GCC_VERSION=$(gcc --version | head -n1)
        log_info "gcc: $GCC_VERSION"
    fi
}

# Check current pywhispercpp installation
check_current_installation() {
    log_step "Checking current pywhispercpp installation..."
    
    if python -c "import pywhispercpp" 2>/dev/null; then
        CURRENT_VERSION=$(python -c "import pywhispercpp; print(pywhispercpp.__version__)" 2>/dev/null || echo "unknown")
        log_info "Current pywhispercpp version: $CURRENT_VERSION"
        
        # Check if CUDA is already enabled
        if python -c "from utils.cuda_utils import CUDAManager; cm = CUDAManager(); print(cm._check_pywhispercpp_cuda())" 2>/dev/null | grep -q "True"; then
            log_warn "pywhispercpp already has CUDA support!"
            if [ "$FORCE_MODE" = false ]; then
                read -p "Reinstall anyway? (y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    log_info "Build cancelled by user"
                    exit 0
                fi
            fi
        fi
        
        # Backup current installation
        log_info "Backing up current installation to: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
        
        # Get site-packages path
        SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
        
        if [ -d "$SITE_PACKAGES/pywhispercpp" ]; then
            cp -r "$SITE_PACKAGES/pywhispercpp" "$BACKUP_DIR/"
            log_info "Backed up pywhispercpp module"
        fi
        
        # Also backup .dist-info if it exists
        DIST_INFO=$(find "$SITE_PACKAGES" -maxdepth 1 -name "pywhispercpp*.dist-info" -type d 2>/dev/null | head -n1)
        if [ -n "$DIST_INFO" ]; then
            cp -r "$DIST_INFO" "$BACKUP_DIR/"
            log_info "Backed up pywhispercpp metadata"
        fi
        
    else
        log_info "No existing pywhispercpp installation found"
    fi
}

# Set up build environment
setup_build_env() {
    log_step "Setting up build environment..."
    
    # Export CUDA paths
    # NOTE: Do not prepend /usr/bin (or CUDA bin) to PATH because it can shadow
    # conda/venv binaries (e.g., pip) and break the build.
    if [ -n "$NVCC_PATH" ] && [ -x "$NVCC_PATH" ]; then
        NVCC_DIR="$(dirname "$NVCC_PATH")"
        case ":$PATH:" in
            *":$NVCC_DIR:"*) : ;;
            *) export PATH="$PATH:$NVCC_DIR" ;;
        esac
    else
        case ":$PATH:" in
            *":$CUDA_HOME/bin:"*) : ;;
            *) export PATH="$PATH:$CUDA_HOME/bin" ;;
        esac
    fi
    
    if [ -d "$CUDA_HOME/lib64" ]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    elif [ -d "$CUDA_HOME/lib" ]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
    fi
    
    # Set CMake arguments for CUDA
    if [ -n "$NVCC_PATH" ] && [ -x "$NVCC_PATH" ]; then
        export CMAKE_ARGS="-DGGML_CUDA=1 -DCMAKE_CUDA_COMPILER=$NVCC_PATH"
    else
        export CMAKE_ARGS="-DGGML_CUDA=1 -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"
    fi
    export GGML_CUDA=1
    
    log_info "Build environment configured:"
    if [ -n "$NVCC_PATH" ]; then
        log_info "  NVCC_PATH: $NVCC_PATH"
    fi
    log_info "  CUDA_HOME: $CUDA_HOME"
    log_info "  LD_LIBRARY_PATH includes: $CUDA_HOME/lib64"
    log_info "  CMAKE_ARGS: $CMAKE_ARGS"
    log_info "  GGML_CUDA: $GGML_CUDA"
    
    # Clean up any previous build artifacts
    log_info "Cleaning previous build artifacts..."
    rm -rf /tmp/pip-build-* 2>/dev/null || true
    rm -rf ~/.cache/pip/wheels/*pywhispercpp* 2>/dev/null || true
}

# Build pywhispercpp with CUDA
build_pywhispercpp() {
    log_step "Building pywhispercpp with CUDA support..."
    log_info "This may take 5-15 minutes depending on your system..."
    echo ""
    
    # Determine verbosity
    PIP_VERBOSE=""
    if [ "$VERBOSE_MODE" = true ]; then
        PIP_VERBOSE="-v"
        log_info "Verbose mode enabled - full build output will be shown"
    fi
    
    # Build command
    BUILD_CMD="pip install git+https://github.com/absadiki/pywhispercpp --no-cache-dir --force-reinstall $PIP_VERBOSE"
    
    log_info "Running: $BUILD_CMD"
    echo ""
    
    # Run build with logging
    if [ "$VERBOSE_MODE" = true ]; then
        # Show output in real-time
        eval "$BUILD_CMD" 2>&1 | tee -a "$LOG_FILE"
        BUILD_EXIT_CODE=${PIPESTATUS[0]}
    else
        # Show progress but less output
        eval "$BUILD_CMD" 2>&1 | tee -a "$LOG_FILE" | grep -E "(Building|Compiling|Creating|Successfully|error|Error|ERROR)" || true
        BUILD_EXIT_CODE=${PIPESTATUS[0]}
    fi
    
    if [ $BUILD_EXIT_CODE -ne 0 ]; then
        log_error "Build failed with exit code $BUILD_EXIT_CODE"
        log_error "See full log at: $LOG_FILE"
        return 1
    fi
    
    log_info "Build completed successfully!"
    return 0
}

# Verify the installation
verify_installation() {
    log_step "Verifying installation..."
    
    # Test import
    if python -c "import pywhispercpp; print('Import successful')" 2>&1 | grep -q "Import successful"; then
        log_info "✓ pywhispercpp imports successfully"
    else
        log_error "✗ Failed to import pywhispercpp"
        return 1
    fi
    
    # Check version
    NEW_VERSION=$(python -c "import pywhispercpp; print(pywhispercpp.__version__)" 2>/dev/null || echo "unknown")
    log_info "Installed version: $NEW_VERSION"
    
    # Check CUDA support
    log_info "Checking CUDA support in pywhispercpp..."
    
    # Try to run the CUDA validation
    CUDA_CHECK_OUTPUT=$(python -c "
import sys
sys.path.insert(0, '.')
try:
    from utils.cuda_utils import CUDAManager
    cm = CUDAManager()
    is_valid, message = cm.validate_cuda_setup(verbose=False)
    if is_valid:
        print('CUDA_READY')
    else:
        print(f'CUDA_NOT_READY: {message}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if echo "$CUDA_CHECK_OUTPUT" | grep -q "CUDA_READY"; then
        log_info "✓ CUDA support verified - GPU acceleration is ready!"
        return 0
    elif echo "$CUDA_CHECK_OUTPUT" | grep -q "CUDA_NOT_READY"; then
        CUDA_MESSAGE=$(echo "$CUDA_CHECK_OUTPUT" | grep "CUDA_NOT_READY" | sed 's/CUDA_NOT_READY: //')
        log_warn "⚠ Installation completed but CUDA not fully ready:"
        log_warn "  $CUDA_MESSAGE"
        log_warn "You may need to restart your Python environment or check CUDA setup"
        return 0  # Don't fail, but warn
    else
        log_warn "⚠ Could not verify CUDA support automatically"
        log_warn "  Output: $CUDA_CHECK_OUTPUT"
        return 0  # Don't fail
    fi
}

# Print troubleshooting information
print_troubleshooting() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              Troubleshooting Information                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    log_info "Build log saved to: $LOG_FILE"
    
    if [ -d "$BACKUP_DIR" ]; then
        log_info "Backup saved to: $BACKUP_DIR"
    fi
    
    echo ""
    echo "Common issues and solutions:"
    echo ""
    echo "1. CUDA not found:"
    echo "   - Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
    echo "   - Set CUDA_HOME: export CUDA_HOME=/usr/local/cuda"
    echo ""
    echo "2. NVCC not found:"
    echo "   - Add CUDA to PATH: export PATH=\$CUDA_HOME/bin:\$PATH"
    echo ""
    echo "3. Build fails with compiler errors:"
    echo "   - Install build tools: sudo apt-get install build-essential"
    echo "   - Make sure cmake version is 3.18 or higher"
    echo ""
    echo "4. pywhispercpp still using CPU after build:"
    echo "   - Check that pywhispercpp was rebuilt with CUDA:"
    echo "     python -c 'from utils.cuda_utils import CUDAManager; CUDAManager().validate_cuda_setup()'"
    echo "   - Restart your Python environment"
    echo "   - Check the build log for CUDA compilation errors"
    echo ""
    echo "5. GPU out of memory:"
    echo "   - Use a smaller model (e.g., 'small' instead of 'large-v3')"
    echo "   - Close other GPU applications"
    echo "   - Set model in config: python -c 'from config.settings import config; config.model.default_model=\"small\"; config.save_config()'"
    echo ""
    echo "For more help, see: CUDA_SETUP.md"
    echo ""
    echo "Known-good command set for this repo/environment:"
    echo "  source \"\$HOME/miniforge3/etc/profile.d/conda.sh\""
    echo "  conda activate py_cripit"
    echo "  export CUDA_HOME=/usr/lib/cuda"
    echo "  bash ./build_cuda.sh --force --verbose"
    echo "  python -c 'from utils.cuda_utils import CUDAManager; print(CUDAManager().validate_cuda_setup())'"
    echo ""
}

# Print success message
print_success() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                  Build Successful!                         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    log_info "pywhispercpp has been built with CUDA support"
    log_info "GPU acceleration is now available for transcription"
    echo ""
    echo "Next steps:"
    echo "  1. Test the installation:"
    echo "     python -c 'from utils.cuda_utils import CUDAManager; print(CUDAManager().validate_cuda_setup())'"
    echo ""
    echo "  2. Run CripIt:"
    echo "     python main.py"
    echo ""
    echo "  3. Check GPU status in the GUI (Status bar will show GPU info)"
    echo ""
    log_info "Build log: $LOG_FILE"
    echo ""
    echo "Command set used / recommended:"
    echo "  source \"\$HOME/miniforge3/etc/profile.d/conda.sh\""
    echo "  conda activate py_cripit"
    echo "  export CUDA_HOME=/usr/lib/cuda"
    echo "  bash ./build_cuda.sh --force --verbose"
    echo ""
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        echo ""
        log_error "Build failed! Check the log for details: $LOG_FILE"
        print_troubleshooting
    fi
}

# Set trap to call cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    # Initialize log file
    echo "Build started at $(date)" > "$LOG_FILE"
    
    print_header
    
    # Run all checks
    check_conda_env
    check_python
    check_cuda
    check_build_tools
    
    # Check current installation
    check_current_installation
    
    # Confirm build
    if [ "$FORCE_MODE" = false ]; then
        echo ""
        echo "Ready to build pywhispercpp with CUDA support."
        echo "This will take 5-15 minutes and requires internet access."
        echo ""
        read -p "Proceed with build? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Build cancelled by user"
            exit 0
        fi
    fi
    
    # Setup and build
    setup_build_env
    
    if build_pywhispercpp; then
        # Verify the build
        verify_installation
        
        # Success!
        print_success
        
        # Restore trap
        trap - EXIT
        exit 0
    else
        # Build failed - cleanup will print troubleshooting
        exit 1
    fi
}

# Run main function
main
