# ARC runner setup

This directory contains the ARC configuration and bootstrap script for a local Ubuntu runner node.

The setup script detects host-specific values by default:

- repository URL from the git remote, unless `--repo-url` is set
- Kubernetes node name from the cluster, falling back to `hostname`, unless `--node-name` is set
- runner label from host OS/version/hostname, unless `--runner-label` is set
- runner release from the runner label, unless `--runner-release` is set
- max runner count from physical CPU cores and memory, unless `--max-runners` is set
- cache root from the runner label, unless `--cache-root` is set

Each runner pod defaults to `4` CPUs and `16Gi` memory. Override those with `--runner-cpu` and `--runner-memory`.

## Files

- [Dockerfile.arc-runner-ubuntu-24.04](Dockerfile.arc-runner-ubuntu-24.04) builds the Ubuntu 24.04 runner image with common build tools, GitHub CLI, `clang`, `clang-tidy`, and passwordless sudo for the `runner` user.
- [arc-runner-entrypoint.sh](arc-runner-entrypoint.sh) aligns the container `runner` UID/GID with the host cache owner before starting the Actions runner.
- [arc-runner-cleanup.sh](arc-runner-cleanup.sh) restores cache ownership on shutdown.
- [arc-xpu-ubuntu-24.04-values.yaml](arc-xpu-ubuntu-24.04-values.yaml) is the Helm values file for the runner scale set.
- [setup-arc-runner.sh](setup-arc-runner.sh) installs and deploys everything on a stock Ubuntu host.
- [github-token.secret.example.yaml](github-token.secret.example.yaml) documents the Kubernetes secret shape only. Do not put a real token in git.

## Quick setup on stock Ubuntu

Use this path on a fresh Ubuntu 24.04 node. The script installs base packages, Docker, a local Kubernetes runtime, Helm, ARC, the GitHub token secret, the local runner image, cache directories, and the runner scale set.

```bash
cd .ci/arc
./setup-arc-runner.sh --github-token '<github-token>'
```

The script never prints the token. Passing a secret as a command argument can still expose it briefly through shell history or process listings, so use a short-lived token and clear shell history according to local policy.

To recreate the current `intel/torch-xpu-ops` scale set and labels, pass those values explicitly:

```bash
./setup-arc-runner.sh \
  --github-token '<github-token>' \
  --repo-url 'https://github.com/intel/torch-xpu-ops' \
  --runner-label 'xpu-ubuntu-24.04' \
  --runner-release 'xpu-ubuntu-24-04' \
  --image 'arc-xpu-runner:ubuntu-24.04-tools' \
  --extra-labels 'ubuntu-24.04,ubuntu-latest' \
  --max-runners 12 \
  --min-runners 0 \
  --runner-cpu 4 \
  --runner-memory 16Gi \
  --reserve-cpu 4 \
  --cache-root '/var/cache/arc/xpu-ubuntu-24.04'
```

The GitHub token must be able to register repository runners for the target repository.

The `xpu-ubuntu-24.04` label comes from the ARC `runnerScaleSetName`. The `ubuntu-24.04` and `ubuntu-latest` labels are optional compatibility labels advertised through `scaleSetLabels`.

This runner image is intentionally docker-less. Jobs that need Docker must use a Docker-capable runner label, such as `build`, instead of the generic Ubuntu labels routed to this ARC scale set.

Jobs using `actions/setup-python` may populate `/opt/hostedtoolcache` on first use. Keep network egress to GitHub tool-cache manifests and downloads available, or prewarm the mounted tool cache before relying on those jobs.

## Manual setup

The manual flow below mirrors the script and is useful when you need to audit or customize each step.

### 1. Install Ubuntu packages

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  docker.io \
  gnupg \
  jq \
  tar
```

### 2. Install local Kubernetes

The current local setup uses K3s as a lightweight Kubernetes distribution on stock Ubuntu.

```bash
curl -sfL https://get.k3s.io | sudo sh -s - --write-kubeconfig-mode 0644
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl get nodes
```

If you use the script, the node selector is generated from the detected Kubernetes node name or `--node-name`. If you deploy the static values file manually, set the node selector to your node name before deploying:

```yaml
nodeSelector:
  kubernetes.io/hostname: <your-node-name>
```

### 3. Install Helm

```bash
curl -fsSL https://get.helm.sh/helm-v3.21.3-linux-amd64.tar.gz -o /tmp/helm.tar.gz
tar -xzf /tmp/helm.tar.gz -C /tmp
sudo install -m 0755 /tmp/linux-amd64/helm /usr/local/bin/helm
helm version
```

### 4. Install the ARC controller

```bash
helm upgrade --install arc \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller \
  --namespace arc-systems \
  --create-namespace \
  --timeout 10m
```

### 5. Create the GitHub token secret

Create the namespace and secret. The token value should only be provided at the terminal prompt or through a local environment variable; never commit it.

```bash
kubectl create namespace arc-runners --dry-run=client -o yaml | kubectl apply -f -
read -r -s -p 'GitHub token: ' GITHUB_TOKEN
printf '\n'
kubectl create secret generic github-token \
  --namespace arc-runners \
  --from-literal=github_token="${GITHUB_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -
unset GITHUB_TOKEN
```

### 6. Build and import the runner image

```bash
RUNNER_IMAGE='arc-xpu-runner:ubuntu-24.04-tools'

docker build \
  -t "${RUNNER_IMAGE}" \
  -f Dockerfile.arc-runner-ubuntu-24.04 \
  .

docker save -o /tmp/arc-xpu-runner.tar "${RUNNER_IMAGE}"
sudo k3s ctr images import /tmp/arc-xpu-runner.tar
rm -f /tmp/arc-xpu-runner.tar
```

### 7. Prepare host cache directories

```bash
CACHE_ROOT='/var/cache/arc/xpu-ubuntu-24.04'

sudo mkdir -p \
  "${CACHE_ROOT}/home-cache" \
  "${CACHE_ROOT}/tool-cache"
sudo chown -R "$(id -u):$(id -g)" "${CACHE_ROOT}"
```

The runner mounts these directories at `/home/runner/.cache` and `/opt/hostedtoolcache`.

### 8. Deploy the runner scale set

```bash
helm upgrade --install xpu-ubuntu-24-04 \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --create-namespace \
  -f arc-xpu-ubuntu-24.04-values.yaml \
  --set controllerServiceAccount.name=arc-gha-rs-controller \
  --set controllerServiceAccount.namespace=arc-systems \
  --timeout 10m
```

### 9. Verify ARC state

These commands print only non-secret Kubernetes state.

```bash
kubectl -n arc-systems get pods -o wide
kubectl -n arc-runners get autoscalingrunnersets,autoscalinglisteners,ephemeralrunnersets,ephemeralrunners,pods -o wide
```

The runner scale set should show the configured min/max runner values, and jobs that use one of the configured labels should create runner pods on demand.

## Workflow usage

Use the dedicated label for jobs that must run on this node:

```yaml
runs-on: xpu-ubuntu-24.04
```

The scale set also advertises `ubuntu-24.04` and `ubuntu-latest`, so jobs using those labels can also be routed here.

## Updating the deployment

After changing [Dockerfile.arc-runner-ubuntu-24.04](Dockerfile.arc-runner-ubuntu-24.04), [arc-runner-entrypoint.sh](arc-runner-entrypoint.sh), or [arc-runner-cleanup.sh](arc-runner-cleanup.sh), rebuild and import the image, then recycle any idle runner pods.

```bash
docker build -t arc-xpu-runner:ubuntu-24.04-tools -f Dockerfile.arc-runner-ubuntu-24.04 .
docker save -o /tmp/arc-xpu-runner.tar arc-xpu-runner:ubuntu-24.04-tools
sudo k3s ctr images import /tmp/arc-xpu-runner.tar
rm -f /tmp/arc-xpu-runner.tar
kubectl -n arc-runners delete pod -l actions.github.com/scale-set-name=xpu-ubuntu-24.04 --ignore-not-found
```

After changing [arc-xpu-ubuntu-24.04-values.yaml](arc-xpu-ubuntu-24.04-values.yaml), redeploy the scale set:

```bash
helm upgrade --install xpu-ubuntu-24-04 \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --create-namespace \
  -f arc-xpu-ubuntu-24.04-values.yaml \
  --set controllerServiceAccount.name=arc-gha-rs-controller \
  --set controllerServiceAccount.namespace=arc-systems \
  --timeout 10m
```

## Removal

Remove the runner scale set before deleting the token secret so ARC finalizers can clean up with GitHub.

```bash
helm uninstall xpu-ubuntu-24-04 -n arc-runners
helm uninstall arc -n arc-systems
kubectl delete namespace arc-runners arc-systems
```
