import sys
import pandas as pd
from statistics import mean
from extract_wiki import CalFreq, FeatureProcess, process_list, cal_freq_from_df, cal_score_from_df

sents = [line.strip() for line in open(sys.argv[1]).readlines()]
pairs = [pair.strip().split('\t') for pair in open(sys.argv[2]).readlines()]
pairs = [str((0, *pair)) for pair in pairs]
cal_freq = CalFreq(sys.argv[3])
fp = FeatureProcess(sys.argv[4])
c = process_list(sents, pairs, fp.batched_feature_process)
data = pd.DataFrame(c)
data = cal_freq_from_df(data, cal_freq)
data = cal_score_from_df(data)
print(mean([item['score'] for item in data.to_dict('records')]))