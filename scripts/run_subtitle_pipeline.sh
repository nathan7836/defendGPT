#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATA_ROOT="$PROJECT_ROOT/data"
DATA_CLEAN_ROOT="$PROJECT_ROOT/data_clean"
SRC_DIR="$DATA_ROOT/youtube-subtitile"
PLAIN_DIR="$DATA_CLEAN_ROOT/youtube"

TRAIN_STEPS="${TRAIN_STEPS:-300000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
TRAIN_EMBED_DIM="${TRAIN_EMBED_DIM:-256}"
TRAIN_NUM_HEADS="${TRAIN_NUM_HEADS:-8}"
TRAIN_NUM_LAYERS="${TRAIN_NUM_LAYERS:-12}"
TRAIN_FF_DIM="${TRAIN_FF_DIM:-1024}"
TRAIN_DROPOUT="${TRAIN_DROPOUT:-0.1}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-2}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.98}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-50}"
METRICS_LOG_FRACTION="${METRICS_LOG_FRACTION:-0.05}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-100}"
SAMPLE_PROMPT="${SAMPLE_PROMPT:-Bonjour je suis }"
SAMPLE_MAX_NEW_TOKENS="${SAMPLE_MAX_NEW_TOKENS:-60}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.8}"
SAMPLE_TOP_K="${SAMPLE_TOP_K:-50}"
RUN_NAME="${RUN_NAME:-run-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/trained_models/runs}"
DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
DASHBOARD_DEVICE="${DASHBOARD_DEVICE:-cpu}"
START_DASHBOARD="${START_DASHBOARD:-0}"
EXTRA_DATA_DIRS_SETTING="${EXTRA_DATA_DIRS:-auto}"

LOG_DIR_LOWER=$(printf '%s' "$LOG_DIR" | tr '[:upper:]' '[:lower:]')
if [[ "$LOG_DIR_LOWER" == "none" ]]; then
  LOG_DIR=""
fi

HAVE_RG=false
if command -v rg >/dev/null 2>&1; then
  HAVE_RG=true
else
  printf 'Info : utilitaire rg non détecté, utilisation de find/grep.\n' >&2
fi

count_matching_files() {
  local glob="$1"
  local dir="$2"
  local count
  if [ "$HAVE_RG" = true ]; then
    count=$(rg --files --iglob "$glob" "$dir" 2>/dev/null | wc -l | tr -d ' \t')
  else
    count=$(find "$dir" -type f -iname "$glob" 2>/dev/null | wc -l | tr -d ' \t')
  fi
  printf '%s' "$count"
}

search_pattern() {
  local pattern="$1"
  local dir="$2"
  if [ "$HAVE_RG" = true ]; then
    rg "$pattern" "$dir" >/dev/null
  else
    grep -R -E "$pattern" "$dir" >/dev/null 2>&1
  fi
}

step() {
  printf '\n[step] %s\n' "$1"
}

step "Nettoyage des fichiers VTT"
python3 "$PROJECT_ROOT/scripts/clean_vtt.py"

src_count=$(count_matching_files '*.vtt' "$SRC_DIR")
plain_count=$(count_matching_files '*.txt' "$PLAIN_DIR")

printf 'Fichiers source : %s (.vtt)\n' "$src_count"
printf 'Fichiers nettoyés : %s (.txt)\n' "$plain_count"

if [[ "$plain_count" != "$src_count" ]]; then
  printf 'Avertissement : nombre de transcriptions (.txt) différent des sous-titres (.vtt).\n' >&2
fi

step "Contrôles rapides sur le texte nettoyé"
if search_pattern "WEBVTT" "$PLAIN_DIR"; then
  printf 'Metadonnées WEBVTT restantes détectées.\n' >&2
else
  printf 'OK : aucune trace WEBVTT.\n'
fi

if search_pattern "<[^>]+>" "$PLAIN_DIR"; then
  printf 'Balises HTML encore présentes.\n' >&2
else
  printf 'OK : aucune balise HTML détectée.\n'
fi

step "Entraînement du mini-transformer"
printf 'Configuration : steps=%s, batch=%s, block=%s, device=%s, embed=%s, layers=%s, heads=%s, lr=%s\n' \
  "$TRAIN_STEPS" "$BATCH_SIZE" "$BLOCK_SIZE" "$DEVICE" \
  "$TRAIN_EMBED_DIM" "$TRAIN_NUM_LAYERS" "$TRAIN_NUM_HEADS" "$TRAIN_LR"

python_args=(
  "$PROJECT_ROOT/scripts/train_subtitles_transformer.py"
  --max-steps "$TRAIN_STEPS"
  --eval-interval "$EVAL_INTERVAL"
  --batch-size "$BATCH_SIZE"
  --block-size "$BLOCK_SIZE"
  --device "$DEVICE"
  --embed-dim "$TRAIN_EMBED_DIM"
  --num-heads "$TRAIN_NUM_HEADS"
  --num-layers "$TRAIN_NUM_LAYERS"
  --ff-hidden-dim "$TRAIN_FF_DIM"
  --dropout "$TRAIN_DROPOUT"
  --lr "$TRAIN_LR"
  --weight-decay "$TRAIN_WEIGHT_DECAY"
  --train-split "$TRAIN_SPLIT"
  --eval-batches "$TRAIN_EVAL_BATCHES"
  --sample-interval "$SAMPLE_INTERVAL"
  --sample-prompt "$SAMPLE_PROMPT"
  --sample-max-new-tokens "$SAMPLE_MAX_NEW_TOKENS"
  --sample-temperature "$SAMPLE_TEMPERATURE"
  --sample-top-k "$SAMPLE_TOP_K"
  --metrics-log-fraction "$METRICS_LOG_FRACTION"
)

collect_auto_extra_dirs() {
  local base="$DATA_CLEAN_ROOT"
  while IFS= read -r -d '' dir; do
    extras+=("$dir")
  done < <(find "$base" -maxdepth 1 -mindepth 1 -type d ! -name 'youtube' ! -name '_history' -print0)
}

extras=()
if [[ "$EXTRA_DATA_DIRS_SETTING" == "auto" ]]; then
  collect_auto_extra_dirs
elif [[ -n "$EXTRA_DATA_DIRS_SETTING" ]]; then
  # Space or newline separated list supplied by user.
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    extras+=("$entry")
  done < <(printf '%s' "$EXTRA_DATA_DIRS_SETTING" | tr ' ' '\n')
fi

if (( ${#extras[@]} > 0 )); then
  printf 'Sources additionnelles détectées :\n'
  for extra in "${extras[@]}"; do
    printf '  - %s\n' "$extra"
    python_args+=(--extra-data-dir "$extra")
  done
else
  printf 'Aucune source additionnelle détectée pour l\'entraînement.\n'
fi

if [[ -n "$RUN_NAME" ]]; then
  python_args+=(--run-name "$RUN_NAME")
fi

if [[ -n "$LOG_DIR" ]]; then
  python_args+=(--log-dir "$LOG_DIR")
fi

python3 "${python_args[@]}"

if [[ -n "$LOG_DIR" ]]; then
  RUN_DIR="$LOG_DIR/$RUN_NAME"
else
  RUN_DIR=""
fi

printf "\nPipeline terminé. Modèle sauvegardé dans %s/trained_models\n" "$PROJECT_ROOT"
if [[ -n "$RUN_DIR" ]]; then
  printf 'Résumé : run=%s | logs=%s\n' "$RUN_NAME" "$RUN_DIR"
  if [[ "$START_DASHBOARD" != "1" ]]; then
    printf 'Dashboard manuel : python dashboard/server.py --run-dir "%s" --checkpoint "%s/checkpoint.pt" --device %s --port %s\n' \
      "$RUN_DIR" "$RUN_DIR" "$DASHBOARD_DEVICE" "$DASHBOARD_PORT"
  fi
else
  printf 'Résumé : run=%s | logs=disabled\n' "$RUN_NAME"
fi

if [[ "$START_DASHBOARD" == "1" && -n "$RUN_DIR" ]]; then
  step "Lancement du dashboard web"
  if [[ ! -f "$RUN_DIR/checkpoint.pt" ]]; then
    printf 'Avertissement : checkpoint introuvable dans %s, dashboard non démarré.\n' "$RUN_DIR" >&2
  else
    printf 'Dashboard sur http://%s:%s (Ctrl-C pour arrêter)\n' "$DASHBOARD_HOST" "$DASHBOARD_PORT"
    python3 "$PROJECT_ROOT/dashboard/server.py" \
      --run-dir "$RUN_DIR" \
      --checkpoint "$RUN_DIR/checkpoint.pt" \
      --host "$DASHBOARD_HOST" \
      --port "$DASHBOARD_PORT" \
      --device "$DASHBOARD_DEVICE"
  fi
fi
