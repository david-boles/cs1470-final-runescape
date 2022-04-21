#!/bin/bash

# Activate the venv, creating it beforehand if missing and
# installing requirements if newly created or given the
# "reinstall" argument.



# Check that this script is being sourced
(return 0 2>/dev/null) && sourced=1 || sourced=0
if [ $sourced -eq 0 ]; then
    echo "Run this script with either \`source enter_venv.sh\` or \`. enter_venv.sh\`!"
    exit 1
fi



# Flag for whether requirements will need to be installed
do_install=0 
if [ "$1" = "reinstall" ]; then
    do_install=1
fi
if [ ! -d "env" ]; then
    do_install=1
fi


# Create venv, cleaning up and gently failing if something goes wrong
if [ ! -d "env" ]; then 
    echo "Creating new venv..."
    (
        # Bash safety checks, not all necessary but might as well keep them
        set -e
        set -u
        set -o pipefail

        python3 -m pip install virtualenv
        virtualenv -p python3 env
    ) || { printf "\n\nWhoops! An error occured while creating the venv. See above ^\n"; rm -rf env; return; }
fi



# Activate venv
source env/bin/activate



# Install requirements, genly failing if something goes wrong
if [ $do_install -eq 1 ]; then
    echo "Installing requirements..."
    (
        # Bash safety checks, not all necessary but might as well keep them
        set -e
        set -u
        set -o pipefail
        
        pip3 install -U pip
        pip3 install -r requirements.txt
    ) || { printf "\n\nWhoops! An error occured while installing requirements. See above ^\nRun \`source enter_venv.sh reinstall\` to try again.\n" && return; }
fi