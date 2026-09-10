# deploy notes

## Flow Matching submodule (`third_party/fm`)

`.gitmodules` pins this repo:

```
path   = third_party/fm
url    = https://github.com/zangyujie2004/flow_matching.git
branch = policy
```

Deploy does not auto-follow remote. The parent repo records a **commit SHA**.
Updating the branch on GitHub does nothing until you fetch and commit a new pin.

### First clone / after clone

```bash
cd deploy
git submodule update --init --recursive third_party/fm
```

### Update `fm` to the latest `policy`

```bash
cd deploy
git submodule update --remote --checkout third_party/fm
git -C third_party/fm log -1 --oneline --decorate   # expect policy @ newest SHA
git add .gitmodules third_party/fm
git commit -m "Bump third_party/fm to latest policy."
```

`--remote` fetches the branch named in `.gitmodules` (`policy`) and checks that
commit out. `git add third_party/fm` records the new SHA in deploy.

Equivalent manual form:

```bash
git -C third_party/fm fetch origin policy
git -C third_party/fm checkout policy
git -C third_party/fm pull --ff-only origin policy
git add third_party/fm
git commit -m "Bump third_party/fm to latest policy."
```

### Check what is pinned

```bash
git submodule status                 # space = matches pin; + = dirty vs pin
git -C third_party/fm rev-parse --abbrev-ref HEAD
git -C third_party/fm log -1 --oneline
```

If HEAD is detached, that is normal: the parent pin is a SHA, not a live branch.
To follow `policy` locally: `git -C third_party/fm checkout policy`.
