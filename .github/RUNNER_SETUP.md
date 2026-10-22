# Self-hosted runner setup (macOS arm64)

This repo is public, so only run the runner for trusted workflows. `ci.yml` has no `pull_request` trigger on purpose: a fork PR can edit the workflow and run on this machine.

1. Repo → Settings → Actions → Runners → **New self-hosted runner** → macOS / ARM64.
   Follow the download and `./config.sh --url ... --token ...` steps in `~/actions-runner`.
   Keep the default labels (`self-hosted`, `macOS`, `ARM64`).
2. Install as a launchd service: `./svc.sh install && ./svc.sh start`.
3. The service does not load shell profiles. Add the tool directories to `~/actions-runner/.path`
   (one `PATH` line, then `./svc.sh stop && ./svc.sh start`):

   ```
   /Users/<you>/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin
   ```

4. Install the tools the workflow calls directly: `brew install gitleaks actionlint jq`
   (`uv` provides `uvx ruff` and `uvx pip-audit`).
5. Repo → Settings → Actions → General → Fork pull request workflows →
   **Require approval for all outside collaborators**.
