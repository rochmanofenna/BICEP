#!/usr/bin/env bash
set -euo pipefail

#Create & activate virtualenv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

#Install requirements
echo "Installing Python dependencies…"
pip install --upgrade pip
pip install -r requirements.txt

#Run the Brownian-motion stage
echo "Generating Brownian paths…"
python -m src.randomness.brownian_motion

#Run the graph-walk stage
echo "Running graph walks…"
python -m src.randomness.brownian_graph_walk

echo "Pipeline finished executing!"
