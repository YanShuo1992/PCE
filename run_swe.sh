#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SECRETS="secrets.env"

load_secrets() {
    if [[ -f "$SECRETS" ]]; then
        set -a; . "$SECRETS"; set +a
    else
        echo "no $SECRETS -- DEEPSEEK_API_KEY must already be exported" >&2
    fi
}

case "${1:-}" in


distill)
    load_secrets
    shift
    exec python run.py distill "$@"
    ;;


gea)
    load_secrets
    shift
    exec python run.py gea "$@"
    ;;



show)
    load_secrets
    case "${2:-}" in
        distill|gea) arm="$2"; shift 2; exec python run.py "$arm" --dry "$@" ;;
        *) echo "usage: $0 show {distill|gea}" >&2; exit 2 ;;
    esac
    ;;

*)
    echo "usage: $0 {distill|gea} [run.py args]" >&2
    echo "       $0 show <arm>" >&2
    exit 2
    ;;
esac
