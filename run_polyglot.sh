#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

load_secrets() {
    if [ -f secrets.env ]; then set -a; . secrets.env; set +a; fi
}

case "${1:-}" in


show)
    case "${2:-}" in
        polyglot_gea|polyglot_distill) load_secrets; arm="$2"; shift 2; exec python run.py "$arm" --dry "$@" ;;
        *) echo "usage: $0 show {polyglot_gea|polyglot_distill}" >&2; exit 2 ;;
    esac
    ;;

images)
    exec bash polyglot/rebuild_images.sh
    ;;

polyglot_gea)
    load_secrets
    shift
    exec python run.py polyglot_gea "$@"
    ;;

polyglot_distill)
    load_secrets
    shift
    exec python run.py polyglot_distill "$@"
    ;;


*)
    echo "usage: $0 {polyglot_gea|polyglot_distill} [run.py args]" >&2
    echo "       $0 show <arm>" >&2
    echo "       $0 images" >&2
    exit 2
    ;;
esac
