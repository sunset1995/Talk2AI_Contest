#!/bin/bash

problem_path="$(pwd -P)/$1"
answer_path="$(pwd -P)/$2"

if [[ "$1" = /* ]]; then
    problem_path="$1"
fi

if [[ "$2" = /* ]]; then
    answer_path="$2"
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

python3 architecture_1.py $problem_path
python3 architecture_2.py $problem_path
python3 naive.py $problem_path
python3 voter.py $answer_path 
