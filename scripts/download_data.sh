#!/usr/bin/env bash
# Download the three public UCI datasets into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/../data/raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data/raw"; cd "$(dirname "$0")/../data/raw"; }
U=https://archive.ics.uci.edu/ml/machine-learning-databases
curl -sSfO $U/communities/communities.data
curl -sSfO $U/communities/communities.names
curl -sSfO $U/adult/adult.data
curl -sSfO $U/adult/adult.test
curl -sSfL -o diabetes.zip "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
unzip -o -q diabetes.zip diabetic_data.csv && rm diabetes.zip
echo "data/raw ready"
