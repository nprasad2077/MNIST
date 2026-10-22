#!/usr/bin/env bash
# Install pinned CI tools (Linux x64/arm64) on the self-hosted runner and put them on PATH.
# Usage: setup-tools.sh [uv] [gitleaks] [actionlint]
# Needs only curl, tar and sha256sum. Downloads are verified against the published checksums.
set -euo pipefail

UV_VERSION=0.10.7
GITLEAKS_VERSION=8.30.1
ACTIONLINT_VERSION=1.7.12

# Versions are part of the path, so bumping a pin installs into a fresh directory.
BIN="${RUNNER_TOOL_CACHE:-$HOME/.cache}/mnist-ci/uv${UV_VERSION}-gl${GITLEAKS_VERSION}-al${ACTIONLINT_VERSION}/bin"

case "$(uname -m)" in
  x86_64) UV_ARCH=x86_64 GL_ARCH=x64 AL_ARCH=amd64 ;;
  aarch64 | arm64) UV_ARCH=aarch64 GL_ARCH=arm64 AL_ARCH=arm64 ;;
  *) echo "::error::unsupported architecture $(uname -m)"; exit 1 ;;
esac

# Download <base>/<archive>, verify it against <base>/<sums>, extract <members...> into $BIN.
install_tarball() {
  local base=$1 archive=$2 sums=$3 strip=$4
  shift 4
  local tmp
  tmp=$(mktemp -d)
  curl -sSfL --retry 3 -o "$tmp/$archive" "$base/$archive"
  curl -sSfL --retry 3 -o "$tmp/$sums" "$base/$sums"
  (cd "$tmp" && grep -F "  $archive" "$sums" | sha256sum -c -)
  tar -xzf "$tmp/$archive" -C "$tmp"
  mkdir -p "$BIN"
  for member in "$@"; do
    install -m 0755 "$tmp/${strip}${member}" "$BIN/$member"
  done
  rm -rf "$tmp"
}

for tool in "$@"; do
  if [ -x "$BIN/$tool" ]; then
    continue
  fi
  case "$tool" in
    uv)
      triple="uv-${UV_ARCH}-unknown-linux-gnu"
      # uv publishes one .sha256 per archive; reuse the same verification path.
      base="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}"
      tmp=$(mktemp -d)
      curl -sSfL --retry 3 -o "$tmp/$triple.tar.gz" "$base/$triple.tar.gz"
      curl -sSfL --retry 3 -o "$tmp/$triple.tar.gz.sha256" "$base/$triple.tar.gz.sha256"
      (cd "$tmp" && sha256sum -c "$triple.tar.gz.sha256")
      tar -xzf "$tmp/$triple.tar.gz" -C "$tmp"
      mkdir -p "$BIN"
      install -m 0755 "$tmp/$triple/uv" "$tmp/$triple/uvx" "$BIN/"
      rm -rf "$tmp"
      ;;
    gitleaks)
      install_tarball "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}" \
        "gitleaks_${GITLEAKS_VERSION}_linux_${GL_ARCH}.tar.gz" "gitleaks_${GITLEAKS_VERSION}_checksums.txt" "" gitleaks
      ;;
    actionlint)
      install_tarball "https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}" \
        "actionlint_${ACTIONLINT_VERSION}_linux_${AL_ARCH}.tar.gz" "actionlint_${ACTIONLINT_VERSION}_checksums.txt" "" actionlint
      ;;
    *) echo "::error::unknown tool: $tool"; exit 1 ;;
  esac
done

if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$BIN" >> "$GITHUB_PATH"
fi
export PATH="$BIN:$PATH"
for tool in "$@"; do
  case "$tool" in
    gitleaks) "$tool" version ;;
    actionlint) "$tool" -version | head -1 ;;
    *) "$tool" --version ;;
  esac
done
