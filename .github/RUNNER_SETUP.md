# Self-hosted runner setup (Debian container)

This repo is public, so only run the runner for trusted workflows. `ci.yml` has no `pull_request` trigger on purpose: a fork PR can edit the workflow and run on this machine.

The runner is the `myoung34/github-runner:debian-bookworm` image. Jobs use `runs-on: [self-hosted, linux]`, which matches the image's default `self-hosted`, `Linux` and `X64`/`ARM64` labels.
The image has no ruff, uv, gitleaks or actionlint, so each job installs what it needs with `.github/scripts/setup-tools.sh`.
The script downloads pinned releases, verifies their SHA-256 checksums, caches them under `$RUNNER_TOOL_CACHE`, and adds them to `PATH`.
To bump a tool, change the version at the top of the script.
The script needs `curl`, `tar`, `sha256sum` and `install`, which the image provides.

## docker-compose

```yaml
services:
  github-runner:
    image: myoung34/github-runner:debian-bookworm
    restart: always
    environment:
      REPO_URL: "https://github.com/nprasad2077/MNIST"
      RUNNER_TOKEN: "<registration token>" # Settings → Actions → Runners → New self-hosted runner
      RUNNER_NAME: "coolify-debian-runner"
      RUNNER_WORKDIR: "/tmp/runner/work"
    # No docker.sock mount: the jobs don't use Docker, and the socket gives every job root on the host.
```

- The registration token expires after about an hour. Keep it out of git; set it in Coolify's environment variables, not in this file.
- The container filesystem is disposable. When the container is recreated, the next run downloads the tools again.

## Repo setting

Settings → Actions → General → Fork pull request workflows → **Require approval for all outside collaborators**.

Check the container with `docker exec <container> bash -c 'cd "$(mktemp -d)" && curl -sSfL https://raw.githubusercontent.com/nprasad2077/MNIST/dev/.github/scripts/setup-tools.sh | bash -s uv gitleaks actionlint'`.
