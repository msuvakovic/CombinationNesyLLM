#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# local python paths and alfworld data locations
export PYTHONPATH="$PWD/Alfworld:${PYTHONPATH:-}"
export ALFWORLD_DATA="$PWD/Alfworld/data/alfdata"
echo "DEBUG: ALFWORLD_DATA is currently set to: $ALFWORLD_DATA"

echo BASE_DIR
#set api key values for code
GOOGLE_API_KEY=""      # Gemini
OPENAI_API_KEY=""      # OpenAI
REPLICATE_API_TOKEN="" # Meta / LLaMA

# set engine type
ENGINE="gpt-4o"
PROCEDURE="general"

#export api keys only if provided
[[ -n "$GOOGLE_API_KEY" ]] && export GOOGLE_API_KEY
[[ -n "$OPENAI_API_KEY" ]] && export OPENAI_API_KEY
[[ -n "$REPLICATE_API_TOKEN" ]] && export REPLICATE_API_TOKEN


python -m Alfworld.main   --procedure $PROCEDURE   --engine $ENGINE --method ours --grounding --refine