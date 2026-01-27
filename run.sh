#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# local python paths and alfworld data locations
export PYTHONPATH="$PWD/Alfworld${PYTHONPATH:-}"
export ALFWORLD_DATA="$PWD/data/alfworld"

#set api key values for code
GOOGLE_API_KEY=""      # Gemini
OPENAI_API_KEY=""      # OpenAI
REPLICATE_API_TOKEN="" # Meta / LLaMA


#export api keys only if provided
[[ -n "$GOOGLE_API_KEY" ]] && export GOOGLE_API_KEY
[[ -n "$OPENAI_API_KEY" ]] && export OPENAI_API_KEY
[[ -n "$REPLICATE_API_TOKEN" ]] && export REPLICATE_API_TOKEN


python -m Alfworld.main   --procedure general   --engine gemini-1.0-propy
