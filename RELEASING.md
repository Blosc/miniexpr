# Releasing a version

## Preliminaries

- Make sure that `RELEASE_NOTES.md` and `ANNOUNCE.md` are up to date with the
  latest news in the release.
- Check that the version symbols in `src/miniexpr.h` contain the correct info.
- If API/ABI changes, increase the minor number (e.g. 0.1 -> 0.2).
- Commit the changes:

```bash
git commit -a -m "Getting ready for release X.Y.Z"
git push
```

## Testing

Create a new `build/` directory, change into it and issue:

```bash
cmake ..
cmake --build . -j10
ctest
```

## Tagging

- Create a tag `vX.Y.Z` from `main`:

```bash
git tag -a vX.Y.Z -m "Tagging version X.Y.Z"
```

- Push the tag to GitHub:

```bash
git push --tags
```

- Create a new release at https://github.com/Blosc/miniexpr/releases/new and
  paste the release notes from `RELEASE_NOTES.md`.

## Announcing (not really needed yet)

- Send an announcement to the blosc mailing list.
  Use `ANNOUNCE.md` as the skeleton.
- Post on the @Blosc2 account at https://fosstodon.org.

## Post-release actions

- Update the version symbols in `src/miniexpr.h` to the next development
  version (e.g. X.Y.Z -> X.Y.(Z+1).dev) if you track dev versions.
- Add a new placeholder section to `RELEASE_NOTES.md` for the next release.
- Commit the changes:

```bash
git commit -a -m "Post X.Y.Z release actions done"
git push
```

That's all folks!
