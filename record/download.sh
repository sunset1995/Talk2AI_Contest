#!/bin/bash

scp aiuser3603@140.110.9.19:~/tmp_dual_lstm_12 ./
scp aiuser3603@140.110.9.19:~/tmp_dual_lstm_13 ./
scp aiuser3603@140.110.9.19:~/tmp_dual_lstm_14 ./
scp -i ~/.ssh/google-cloud s2821d3721@35.199.165.200:~/Talk2AI_Contest/tmp_* ./
scp -i ~/.ssh/google-cloud s2821d3721@104.198.217.161:~/Talk2AI_Contest/tmp_* ./
scp -i ~/.ssh/google-cloud s2821d3721@35.196.130.216:~/Talk2AI_Contest/tmp_* ./
scp aiuser3603@140.110.9.19:~/tmp_dual_lstm_18 ./

cat tmp_dual_lstm_12 | grep Valid > exp_12
cat tmp_dual_lstm_13 | grep Valid > exp_13
cat tmp_dual_lstm_14 | grep Valid > exp_14
cat tmp_dual_lstm_15 | grep Valid > exp_15
cat tmp_dual_lstm_16 | grep Valid > exp_16
cat tmp_dual_lstm_17 | grep Valid > exp_17
cat tmp_dual_lstm_18 | grep Valid > exp_18
