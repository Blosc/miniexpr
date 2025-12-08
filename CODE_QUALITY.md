# Code Quality Tools

## Setup (For Contributors)

After cloning the repository, install the Git hooks:

```bash
./scripts/install-hooks.sh
```

This will install the pre-commit hook that checks for trailing whitespace.

## Trailing Whitespace

Trailing whitespace is automatically prevented by the pre-commit hook and can be checked/fixed using the provided tools.

### Pre-commit Hook

The Git pre-commit hook automatically checks for trailing whitespace before each commit.

**First-time setup:**
```bash
./scripts/install-hooks.sh
```

If trailing whitespace is detected, the commit will be blocked with instructions on how to fix it.

To bypass the check (not recommended):
```bash
git commit --no-verify
```

### Manual Checking

Check for trailing whitespace:
```bash
make check-whitespace
# or
./check-whitespace.sh
```

Fix trailing whitespace:
```bash
make fix-whitespace
# or
./check-whitespace.sh --fix
```

### Files Checked

The tools check these file types:
- `*.c` - C source files
- `*.h` - C header files
- `*.md` - Markdown documentation

Build artifacts in `build/` and `.git/` are excluded.

### CI Integration

Add to your CI pipeline:
```yaml
- name: Check trailing whitespace
  run: make check-whitespace
```
