#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== [0] safety check: is a polyglot run live? =="
if ps -eo cmd | grep -E "GEA_outer.*--polyglot" | grep -qv grep; then
  echo "!! a polyglot run is live, aborting"; exit 1
fi
echo "   no. (SWE runs are unaffected: they use the dgm / sweb.* namespace)"

echo "== [1] remove orphaned pb.eval containers =="
docker ps -aq --filter name=pb.eval | xargs -r docker rm -f >/dev/null
echo "   done"

echo "== [2] delete every pb.eval image (the name still matches, so a stale one is skipped forever) =="
docker images --format '{{.Repository}}:{{.Tag}}' | grep '^pb\.eval\.' | xargs -r docker rmi -f >/dev/null 2>&1
echo "   pb.eval left after deleting: $(docker images --format '{{.Repository}}' | grep -c '^pb\.eval\.')"
echo "   kept pb.env: $(docker images --format '{{.Repository}}' | grep -c '^pb\.env\.')  pb.base: $(docker images --format '{{.Repository}}' | grep -c '^pb\.base')"

echo "== [3] clear the stale build context (0444 read-only leftovers are where the EPERM false evolutions came from) =="
rm -rf logs/build_images/instances
echo "   done"

echo "== [4] build all 225 pb.eval images =="
python3 - <<'PY'
import json, sys, docker, time
sys.path.insert(0, '.')
from polyglot.docker_build import build_instance_images
d = json.load(open("polyglot/polyglot_benchmark_metadata.json"))
print(f"   dataset: {len(d)} instances")
t0 = time.time()
build_instance_images(docker.from_env(), d, force_rebuild=False, max_workers=8)
print(f"   elapsed {(time.time()-t0)/60:.1f} min")
PY
echo "== done. pb.eval now: $(docker images --format '{{.Repository}}' | grep -c '^pb\.eval\.') =="
