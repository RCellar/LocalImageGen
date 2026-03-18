#!/usr/bin/env bash
# Shared runtime detection — sourced by start.sh, stop.sh, update.sh

detect_runtime() {
    if podman compose version &>/dev/null 2>&1; then
        echo "podman compose"
    elif command -v podman-compose &>/dev/null; then
        echo "podman-compose"
    elif docker compose version &>/dev/null 2>&1; then
        echo "docker compose"
    elif command -v docker-compose &>/dev/null; then
        echo "docker-compose"
    else
        echo ""
    fi
}

require_runtime() {
    local cmd
    cmd=$(detect_runtime)
    if [ -z "$cmd" ]; then
        echo "ERROR: No container compose runtime found. Install one of:"
        echo "  dnf install podman-compose"
        echo "  # or: https://docs.docker.com/compose/install/"
        exit 1
    fi
    echo "$cmd"
}
