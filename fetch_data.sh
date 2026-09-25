#!/usr/bin/env bash
# Downloads the public tau-bench airline + retail data (MIT license, Sierra Research) into data/.
set -euo pipefail
cd "$(dirname "$0")"
TMP=$(mktemp -d)
git clone --depth 1 https://github.com/sierra-research/tau-bench.git "$TMP/tau-bench"
E="$TMP/tau-bench/tau_bench/envs"
mkdir -p data/airline data/retail
cp "$E/airline/data/"{users,reservations,flights}.json data/airline/
cp "$E/airline/wiki.md" data/airline/policy.md
cp "$E/retail/data/"{users,orders}.json data/retail/
cp "$E/retail/wiki.md" data/retail/policy.md
cp "$TMP/tau-bench/LICENSE" data/TAU_BENCH_LICENSE
rm -rf "$TMP"; echo "data ready in data/"
