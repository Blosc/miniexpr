#!/bin/sh
#
# Install Git hooks for miniexpr
# Run this after cloning the repository
#

echo "Installing Git hooks..."

# Install pre-commit hook
if [ -f scripts/pre-commit ]; then
    cp scripts/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    echo "✅ Installed pre-commit hook (checks trailing whitespace)"
else
    echo "❌ Error: scripts/pre-commit not found"
    exit 1
fi

echo ""
echo "Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will automatically check for trailing whitespace."
echo "To bypass: git commit --no-verify (not recommended)"
echo ""
echo "To manually check: make check-whitespace"
echo "To manually fix:   make fix-whitespace"
