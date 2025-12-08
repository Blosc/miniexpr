#!/bin/sh
#
# Script to check for and optionally fix trailing whitespace
# Usage:
#   ./check-whitespace.sh        # Check only
#   ./check-whitespace.sh --fix  # Fix trailing whitespace
#

FIX=0
if [ "$1" = "--fix" ] || [ "$1" = "-f" ]; then
    FIX=1
fi

# Find files with trailing whitespace
FILES=$(find . -type f \( -name '*.c' -o -name '*.h' -o -name '*.md' \) \
    -not -path './build/*' \
    -not -path './.git/*' \
    -exec grep -l ' $' {} \; 2>/dev/null)

if [ -z "$FILES" ]; then
    echo "✅ No trailing whitespace found"
    exit 0
fi

if [ $FIX -eq 1 ]; then
    echo "🔧 Fixing trailing whitespace in:"
    echo "$FILES" | sed 's/^/  /'
    find . -type f \( -name '*.c' -o -name '*.h' -o -name '*.md' \) \
        -not -path './build/*' \
        -not -path './.git/*' \
        -exec sed -i '' 's/[[:space:]]*$//' {} \;
    echo "✅ Fixed!"
    exit 0
else
    echo "❌ Trailing whitespace found in:"
    echo "$FILES" | sed 's/^/  /'
    echo ""
    echo "To fix, run:"
    echo "  $0 --fix"
    exit 1
fi
