#!/bin/bash
"""
MiniGPT AutoTest Runner
======================
Simple wrapper script for running AutoTest with common options.

Usage:
  ./run_autotest.sh                 # Run full automation
  ./run_autotest.sh check           # Quick setup check only
  ./run_autotest.sh preview         # Dry run preview
  ./run_autotest.sh train           # Training step only
  ./run_autotest.sh backend         # Backend testing only
  ./run_autotest.sh frontend        # Frontend testing only
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "autoTest.py" ]; then
    print_error "autoTest.py not found. Make sure you're in the MiniGPT directory."
    exit 1
fi

# Check Python availability
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    print_error "Python not found. Please install Python 3.8+."
    exit 1
fi

# Use python3 if python is not available
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

print_info "Using Python command: $PYTHON_CMD"

# Parse command line argument
case "${1:-full}" in
    "check")
        print_info "Running quick setup check..."
        $PYTHON_CMD quick_setup_check.py
        ;;

    "preview"|"dry-run")
        print_info "Running AutoTest preview (dry run)..."
        $PYTHON_CMD autoTest.py --dry-run
        ;;

    "train"|"training")
        print_info "Running training step only..."
        $PYTHON_CMD autoTest.py --step train
        ;;

    "backend")
        print_info "Running backend testing only..."
        $PYTHON_CMD autoTest.py --step backend
        ;;

    "frontend")
        print_info "Running frontend testing only..."
        $PYTHON_CMD autoTest.py --step frontend
        ;;

    "data")
        print_info "Running data preparation only..."
        $PYTHON_CMD autoTest.py --step data
        ;;

    "system")
        print_info "Running system check only..."
        $PYTHON_CMD autoTest.py --step system
        ;;

    "full"|"")
        print_info "Running full AutoTest pipeline..."
        echo
        print_warning "This will run the complete automation pipeline:"
        echo "  1. System requirements check"
        echo "  2. Data preparation"
        echo "  3. Model training (3 epochs)"
        echo "  4. Backend testing"
        echo "  5. Frontend testing"
        echo "  6. Integration testing"
        echo
        echo "Estimated time: 15-30 minutes"
        echo

        # Ask for confirmation
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Starting full automation..."
            $PYTHON_CMD autoTest.py
        else
            print_info "Cancelled by user"
            exit 0
        fi
        ;;

    "help"|"-h"|"--help")
        echo "MiniGPT AutoTest Runner"
        echo
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  check      Run quick setup check only"
        echo "  preview    Show what would be executed (dry run)"
        echo "  system     Run system requirements check"
        echo "  data       Run data preparation only"
        echo "  train      Run model training only"
        echo "  backend    Run backend testing only"
        echo "  frontend   Run frontend testing only"
        echo "  full       Run complete automation pipeline (default)"
        echo "  help       Show this help message"
        echo
        echo "Examples:"
        echo "  $0                    # Run full automation"
        echo "  $0 check              # Quick setup verification"
        echo "  $0 preview            # Preview what will run"
        echo "  $0 train              # Train model only"
        ;;

    *)
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

# Check exit code and provide feedback
EXIT_CODE=$?
echo

if [ $EXIT_CODE -eq 0 ]; then
    print_success "Command completed successfully!"

    # Provide next steps based on command
    case "${1:-full}" in
        "check")
            print_info "Next: Run '$0 preview' to see what AutoTest will do"
            ;;
        "preview")
            print_info "Next: Run '$0 full' to execute the full automation"
            ;;
        "full"|"")
            print_success "🎉 AutoTest completed! Your MiniGPT system is ready."
            echo
            print_info "You can now:"
            echo "  • Start the application: ./start-all.sh"
            echo "  • Chat with your model: cd backend && python -m minigpt.chat"
            echo "  • Use web interface: http://localhost:3000"
            ;;
    esac
else
    print_error "Command failed with exit code $EXIT_CODE"
    print_info "Check the logs for details:"
    echo "  • autotest.log - Detailed execution log"
    echo "  • autotest_report_*.json - Test results report"
fi

exit $EXIT_CODE