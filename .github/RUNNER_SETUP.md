# Self-hosted runner setup (Debian / Linux)

This repo is public, so only run the runner for trusted workflows. `ci.yml` has no `pull_request` trigger on purpose: a fork PR can edit the workflow and run on this machine.

Jobs use `runs-on: [self-hosted, linux]`, which matches the default labels GitHub gives a Linux runner.
A self-hosted runner has no tool image, so everything the jobs call must be installed on the machine.

1. Repo → Settings → Actions → Runners → **New self-hosted runner** → Linux (pick x64 or ARM64 to match `uname -m`).
   Follow the download and `./config.sh --url ... --token ...` steps in `~/actions-runner`.
   Keep the default labels (`self-hosted`, `Linux`, plus `X64` or `ARM64`).
   Run it as a normal user, not root.
2. Install as a systemd service: `sudo ./svc.sh install && sudo ./svc.sh start`
   (`sudo ./svc.sh install <user>` to run as a specific user).
3. Install the base packages: `sudo apt-get update && sudo apt-get install -y git curl ca-certificates jq tar`
4. Install the tools the workflow calls directly. `uv` provides `uvx ruff` and `uvx pip-audit`, and downloads the Python 3.11 that the hygiene job needs:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh        # installs to ~/.local/bin

   mkdir -p ~/.local/bin && cd "$(mktemp -d)"
   case "$(uname -m)" in x86_64) A=x64 ;; aarch64) A=arm64 ;; esac

   curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_${A}.tar.gz" \
     | tar xz gitleaks && install gitleaks ~/.local/bin/

   curl -sSfL "https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_linux_${A/x64/amd64}.tar.gz" \
     | tar xz actionlint && install actionlint ~/.local/bin/
   ```

   Optional: `sudo apt-get install -y shellcheck`. `actionlint` runs it on `run:` scripts if present.
5. The service does not load shell profiles. Make sure `~/.local/bin` is on the service's PATH by adding this line to
   `~/actions-runner/.env`, then `sudo ./svc.sh stop && sudo ./svc.sh start`:

   ```
   PATH=/home/<you>/.local/bin:/usr/local/bin:/usr/bin:/bin
   ```

6. Repo → Settings → Actions → General → Fork pull request workflows →
   **Require approval for all outside collaborators**.

Check the setup with `sudo -u <you> env -i PATH=/home/<you>/.local/bin:/usr/bin:/bin sh -c 'uvx --version && gitleaks version && actionlint -version && jq --version'`.
