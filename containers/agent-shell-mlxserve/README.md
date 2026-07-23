# agent-shell-mlxserve

The guest image for **mlx-serve's Agent Sandbox** — the Linux VM the assistant's
shell and MCP tools run inside. A deliberately small, glibc-based shell for
agentic development: Node.js + npm, Python 3 + pip, a real Bash shell, a lean set
of everyday CLI tools, `apt` for on-demand tools, and **pre-baked MCP servers**.

> This is the mlx-serve-specific fork of the generic `agent-shell` image (which
> is shared with the `contain` project). Keep them separate: this one adds the
> baked-in MCP servers the Mac App Store build needs (it can't `npx`-fetch them
> at runtime — App Review guideline 2.5.2), and it publishes under its own repo
> so the two images never collide.
>
> The source of truth lives IN the mlx-serve repo (`containers/agent-shell-mlxserve/`)
> so image and app change together — the app's ssh transport depends on what's
> baked in here, and `SandboxSSHTests.testGuestImageSourceBakesDropbearIn` pins
> this Dockerfile against it.

## Pre-baked MCP servers

mlx-serve runs stdio MCP servers *inside* this guest. The Developer ID build can
`npx`-fetch them on first use; the App Store build cannot, so the common ones are
installed here at build time, pinned by version (see the `MCP_*_VERSION` build
args in the Dockerfile):

- `@modelcontextprotocol/server-filesystem`
- `@modelcontextprotocol/server-github`

`npx <name>` then resolves the global install instead of hitting the npm registry.
The user's *own* `apt`/`npm`/`pip` for their project stay live — only agent tools
are frozen.

## sshd (dropbear) — sandbox agent CLI sessions

mlx-serve's Sandbox window runs agent CLIs (pi, hermes) *inside* this guest over
ssh: the guest init starts `dropbear -R -s -p 22` (host keys generated in-guest
on first connection, key-only auth against a host-injected `authorized_keys` —
no passwords, nothing listens beyond the VM's NAT), and the app mirrors it to
`localhost:<port>` on the Mac. `dropbear-bin` is the ~500 KB binaries-only
package (no init scripts — the sandbox init owns startup). The app **probes for
dropbear** and reports a stale cached image when it's missing, so removing it
here strands every sandbox terminal session after the next pull.

## Publishing

```bash
make export    # linux/arm64 rootfs tarball (rootfs.tar.gz) for the MAS bundle
make push REPO=<youruser>/agent-shell-mlxserve   # push the arm64 image (DMG runtime pull)
```

The Developer ID build pulls this image from the registry at runtime; the App
Store build bundles `rootfs.tar.gz` into the `.app`. mlx-serve's default base
image (`ServerOptions.SandboxConfig.baseImage`) points here.

## Baked-in tools

Beyond the language runtimes, the image ships the CLI utilities an agent reaches
for constantly (all via a single `apt` layer, `--no-install-recommends`):

| Tool / package | Provides |
|---|---|
| `git`, `curl`, `ca-certificates` | clone/fetch over TLS |
| `procps` | `ps`, `pgrep`, `pkill`, `top`, `free`, `uptime`, `vmstat`, `watch` |
| `psmisc` | `killall`, `fuser`, `pstree` |
| `ripgrep` | `rg` — fast recursive code search |
| `lsof` | open files / what's listening on a port (pairs with sandbox port maps) |
| `iproute2` | `ss`, `ip` (modern net tooling; the sandbox init uses `ip link`) |
| `less` | pager expected by `git log` / `man` |
| `nano` | small everyday editor (vim is heavier and left out) |
| `dropbear-bin` | `dropbear` sshd — the Sandbox terminal's agent-session transport (see above) |

## Measured size

| Metric | Size |
|---|---|
| Compressed (registry pull/push) | **~93 MB** |
| Unpacked (on disk) | **~275 MB** |

Base: **Debian 13 "trixie"** (current stable) → Python **3.13** + glibc 2.41.
Where the bytes go (unpacked): Node binary ~98 MB (stripped) · Python 3.13 stdlib
~29 MB · npm ~19 MB · Debian base + apt (rest).

## Why these choices

The base OS is **not** where the weight is — the language runtimes are. Choosing
busybox (~1.5 MB) over Alpine (~8 MB) saves ~6 MB while the runtimes cost ~140 MB,
so the base is noise. We optimize the runtimes and the build, not the distro.

- **glibc / `debian:trixie-slim` (Debian 13, latest stable), not musl/Alpine.** On
  musl, most PyPI packages have no prebuilt wheel and compile from source (slow, needs
  a toolchain, bloats the image). glibc means Python wheels and npm native modules
  "just work." trixie over bookworm: ~5 MB smaller, Python 3.13 vs 3.11, longer
  support — and its newer glibc still runs every older `manylinux` wheel.
- **Node.js, not Bun.** Every real (npm-compatible) JS runtime is V8/JSC-based and
  costs ~90 MB — Bun's binary is ~90 MB too, it is *not* smaller than Node. Node
  has the widest ecosystem + agent-training-data support. (Truly tiny engines like
  QuickJS are ~1 MB but drop npm entirely.)
- **Keep `apt`.** For an agentic shell, on-demand `apt-get install` is high value
  and costs almost nothing (we only drop the package *index cache*, not apt itself).
- **"Practical minimum" cleanup**: `--no-install-recommends`, strip the Node binary
  in a throwaway stage (binutils never ships), drop apt lists / docs / man / locales.

### Image size ≠ RAM
These are independent. This image idles at a few MB of RAM; RAM is driven by what
you *run* (Node ~40–60 MB/process, Python ~10 MB, shell ~3 MB), not by image size.

## Usage

```bash
make build     # build (attestations disabled for honest sizes)
make size      # print unpacked + compressed sizes
make test      # verify runtimes + npm/pip installs end-to-end (needs network)
make shell     # drop into bash
```

Inside the container, `pip install` writes to the system env directly
(`PIP_BREAK_SYSTEM_PACKAGES=1`) since the container is disposable — no venv dance.

## Notes / levers to go smaller or safer

- **Baked CLI tools** (`procps`/`psmisc`/`ripgrep`/`lsof`/`iproute2`/`less`/`nano`):
  a lean set that adds only a few MB unpacked (`ripgrep` is the largest, ~4 MB). Drop
  any you don't need from the apt layer to shave those bytes; the runtimes still dominate.
- **Native Python build deps**: not included. If a package needs to compile, add
  `apt-get install -y build-essential python3-dev` at runtime (or in a builder stage).
- **Drop npm** (save ~19 MB) if the agent only *runs* JS and never installs packages.
- **Run as non-root**: an `agent` user exists but is not the default (root keeps
  `apt`/global installs frictionless). Add `USER agent` to the Dockerfile to switch;
  the agent then needs `sudo` (not installed) or pre-baked deps to use `apt`.
- **Node version** defaults to `24-trixie-slim` (current LTS). Override per build:
  `docker build --build-arg NODE_TAG=26-trixie-slim .`. Sizes barely differ across
  majors (~10 MB uncompressed / ~2 MB compressed spread over 22→26); 26 is the heaviest.
- **`libatomic1` is required** for Node **25+** (its binary links `libatomic.so.1`,
  which a bare `debian:*-slim` lacks). It's installed already, and is a no-op
  for 22/24. Drop it only if you pin Node ≤ 24 and want to shave ~10 KB.
- **Pin versions** via the `NODE_TAG` / `DEBIAN_TAG` build args for reproducibility.
