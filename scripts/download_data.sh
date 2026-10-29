#!/usr/bin/env bash
# Download the public datasets (UCI + AHRQ MEPS) into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/../data/raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data/raw"; cd "$(dirname "$0")/../data/raw"; }
U=https://archive.ics.uci.edu/ml/machine-learning-databases
curl -sSfO $U/communities/communities.data
curl -sSfO $U/communities/communities.names
curl -sSfO $U/adult/adult.data
curl -sSfO $U/adult/adult.test
curl -sSfL -o diabetes.zip "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
unzip -o -q diabetes.zip diabetic_data.csv && rm diabetes.zip
# MEPS 2015 full-year consolidated file (AHRQ public use file HC-181, ~13 MB zipped)
curl -sSfL -o meps_h181.zip https://meps.ahrq.gov/mepsweb/data_files/pufs/h181ssp.zip
mkdir -p meps && unzip -o -q meps_h181.zip -d meps && rm meps_h181.zip
echo "data/raw ready"
