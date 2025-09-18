#!/bin/bash
# Build and manage MiniGPT Docker containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

show_help() {
    echo "MiniGPT Docker Management Script"
    echo "================================"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build     Build Docker images"
    echo "  run       Run containers with docker-compose"
    echo "  stop      Stop running containers"
    echo "  logs      Show container logs"
    echo "  clean     Clean up images and volumes"
    echo "  shell     Open shell in container"
    echo "  help      Show this help message"
    echo ""
    echo "Options:"
    echo "  --api-only    Only build/run API container"
    echo "  --trainer     Only build/run trainer container"
    echo "  --rebuild     Force rebuild images"
    echo "  --follow      Follow logs (with logs command)"
    echo ""
    echo "Examples:"
    echo "  $0 build --rebuild"
    echo "  $0 run --api-only"
    echo "  $0 logs --follow"
    echo "  $0 shell minigpt-api"
}

build_images() {
    local rebuild=""
    local api_only=false
    local trainer_only=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --rebuild)
                rebuild="--no-cache"
                shift
                ;;
            --api-only)
                api_only=true
                shift
                ;;
            --trainer)
                trainer_only=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    log_info "Building MiniGPT Docker images..."

    if [[ "$trainer_only" == "true" ]]; then
        log_info "Building trainer image..."
        docker build $rebuild -f Dockerfile.trainer -t minigpt:trainer .
        log_success "Trainer image built successfully"
    elif [[ "$api_only" == "true" ]]; then
        log_info "Building API image..."
        docker build $rebuild -f Dockerfile -t minigpt:api .
        log_success "API image built successfully"
    else
        log_info "Building API image..."
        docker build $rebuild -f Dockerfile -t minigpt:api .
        log_success "API image built successfully"

        log_info "Building trainer image..."
        docker build $rebuild -f Dockerfile.trainer -t minigpt:trainer .
        log_success "Trainer image built successfully"
    fi

    log_success "All images built successfully!"
}

run_containers() {
    local api_only=false
    local trainer_only=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --api-only)
                api_only=true
                shift
                ;;
            --trainer)
                trainer_only=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    log_info "Starting MiniGPT containers..."

    if [[ "$trainer_only" == "true" ]]; then
        docker-compose up -d minigpt-trainer
        log_success "Trainer container started"
    elif [[ "$api_only" == "true" ]]; then
        docker-compose up -d minigpt-api
        log_success "API container started"
    else
        docker-compose up -d
        log_success "All containers started"
    fi

    log_info "Container status:"
    docker-compose ps
}

stop_containers() {
    log_info "Stopping MiniGPT containers..."
    docker-compose down
    log_success "Containers stopped"
}

show_logs() {
    local follow=""
    local service=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --follow|-f)
                follow="-f"
                shift
                ;;
            *)
                if [[ -z "$service" ]]; then
                    service="$1"
                else
                    log_error "Unknown option: $1"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    if [[ -n "$service" ]]; then
        log_info "Showing logs for $service..."
        docker-compose logs $follow "$service"
    else
        log_info "Showing logs for all containers..."
        docker-compose logs $follow
    fi
}

clean_up() {
    log_warning "This will remove all MiniGPT Docker images and volumes!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Stopping containers..."
        docker-compose down -v --remove-orphans

        log_info "Removing images..."
        docker rmi -f minigpt:api minigpt:trainer 2>/dev/null || true

        log_info "Removing unused volumes..."
        docker volume prune -f

        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

open_shell() {
    local container="$1"

    if [[ -z "$container" ]]; then
        container="minigpt-api"
    fi

    log_info "Opening shell in $container..."
    docker-compose exec "$container" /bin/bash
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose is not installed or not in PATH"
    exit 1
fi

# Main command handling
case "${1:-help}" in
    build)
        shift
        build_images "$@"
        ;;
    run)
        shift
        run_containers "$@"
        ;;
    stop)
        stop_containers
        ;;
    logs)
        shift
        show_logs "$@"
        ;;
    clean)
        clean_up
        ;;
    shell)
        shift
        open_shell "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' to see available commands"
        exit 1
        ;;
esac