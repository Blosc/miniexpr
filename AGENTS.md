# Agent Instructions

- Use K&R braces style (opening brace on the same line, closing brace on its own line) for all edits.
- Keep changes minimal and scoped to the request.
- Follow existing code style and formatting (K&R, naming, etc.).
- Prefer `rg` for search and `apply_patch` for single-file edits.
- Run relevant tests when changing code; report any warnings.
- Avoid adding new dependencies without explicit approval.
- Preserve backward compatibility unless requested.
- Document behavior changes in code comments or docs.
- Keep output tables/logs aligned and stable for diffing.
