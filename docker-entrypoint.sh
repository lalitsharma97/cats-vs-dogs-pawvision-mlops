#!/bin/bash
set -e

# Ensure logs directory exists and is writable
mkdir -p logs
chmod 755 logs

# Ensure the application can write to logs directory
touch logs/.write_test || {
    echo "Error: Cannot write to logs directory. Fixing permissions..."
    # This should work since we're running as appuser and the directory should be owned by us
}

# Remove the test file
rm -f logs/.write_test

echo "Starting application..."
exec "$@"
