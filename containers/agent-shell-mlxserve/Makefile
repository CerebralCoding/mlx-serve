IMAGE ?= agent-shell-mlxserve
TAG   ?= latest
# Push target: the repo mlx-serve's Agent Sandbox pulls from at runtime. This is
# the published image, so it defaults to the real namespace (not $(USER), which
# is the local macOS login and won't have push rights on Docker Hub). A forker
# overrides it:  make push REPO=youruser/agent-shell-mlxserve
REPO  ?= ddalcu/agent-shell-mlxserve

.PHONY: build push size shell test clean export

# --provenance/--sbom=false keeps a single clean manifest so `docker images`
# reports honest sizes (attestation manifests otherwise inflate the number).
build:
	docker build --provenance=false --sbom=false -t $(IMAGE):$(TAG) .

# Build for the arm64 sandbox guest (Apple Silicon HVF) and push to a registry
# the sandbox can pull. MUST be arm64 — the sandbox rejects an amd64-only image.
# Then set this repo as the base image in mlx-serve Settings → Agent Sandbox.
push:
	docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
		-t $(REPO):$(TAG) --push .

# Uncompressed on-disk size (platform-specific), plus compressed pull size.
size:
	@printf 'unpacked (on disk) : '; docker image inspect $(IMAGE):$(TAG) --format '{{.Size}}' | awk '{printf "%.0f MB\n", $$1/1000000}'
	@printf 'compressed (pull)  : '; docker save $(IMAGE):$(TAG) | gzip | wc -c | awk '{printf "%.0f MB\n", $$1/1000000}'

# Drop into the shell.
shell:
	docker run --rm -it $(IMAGE):$(TAG)

# Verify runtimes + that the package managers work end to end (needs network).
test:
	docker run --rm $(IMAGE):$(TAG) bash -c '\
		node --version && npm --version && python3 --version && python3 -m pip --version && \
		cd /tmp && npm init -y >/dev/null && npm install is-odd >/dev/null && echo "npm install OK" && \
		pip install cowsay >/dev/null && python3 -c "import cowsay" && echo "pip install OK"'

clean:
	-docker rmi $(IMAGE):$(TAG) $(IMAGE):clean

# Flatten the arm64 image into a single rootfs tarball for the Mac App Store
# bundle. `docker export` gives a flat filesystem (no layers, no whiteouts), so
# the in-process TarReader unpacks it directly. Checksummed; the app verifies it.
#   make export            -> rootfs.tar.gz + rootfs.tar.gz.sha256
export:
	docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
		-t $(IMAGE):export --load .
	docker create --name mlxrootfs $(IMAGE):export
	docker export mlxrootfs | gzip -9 > rootfs.tar.gz
	docker rm mlxrootfs
	shasum -a 256 rootfs.tar.gz | tee rootfs.tar.gz.sha256
	@printf 'rootfs.tar.gz: '; du -h rootfs.tar.gz | cut -f1
