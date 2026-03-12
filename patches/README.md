## TripoSR local changes

This repo uses `TripoSR/` as a git submodule.

If you clone this repo and want the same local edits that existed on the original machine, initialize the submodule and apply:

```bash
git submodule update --init --recursive
git -C TripoSR apply ../patches/TripoSR-local-changes.patch
```

