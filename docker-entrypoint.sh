#!/bin/bash
set -e

echo "Container starting up..."

# Ensure logs directory exists and is writable
mkdir -p logs || echo "Warning: Could not create logs directory"
chmod 755 logs || echo "Warning: Could not set permissions on logs directory"

# Test if we can write to logs directory
if touch logs/.write_test 2>/dev/null; then
    rm -f logs/.write_test
    echo "Logs directory is writable"
else
    echo "Warning: Logs directory is not writable, continuing anyway"
fi

echo "Starting application..."
exec "$@"
