from sklearn.feature_extraction.text import TfidfVectorizer
import bson
import pprint
import json
import pandas as pd
import time
import random


bcolors = ['\033[95m',
    '\033[94m',
    '\033[96m',
    '\033[92m',
     '\033[93m'
    ,'\033[91m'
    ,'\033[0m'
    ,'\033[1m'
    ,'\033[4m']


def changeColor():
    print(random.choice(bcolors))

start_time = time.time();
changeColor()
pp = pprint.PrettyPrinter(indent=4)

builds = open("restarted_builds.bson", "rb")
logs = open("/scratch2/2191051t/restarted_logs.bson", "rb")
jobs = open("/scratch2/2191051t/restarted_jobs.bson", "rb")

builds_gen = bson.decode_file_iter(builds)
logs_gen = bson.decode_file_iter(logs)
jobs_gen = bson.decode_file_iter(jobs)
print("It worked")

build_df = pd.DataFrame(builds_gen)
build_df = build_df.rename(columns={"id": "build_id"})
print(build_df['old'].apply(pd.Series)['job_ids'])
log_df = pd.DataFrame(logs_gen)


job_df = pd.DataFrame(jobs_gen)
result = pd.merge(job_df, log_df, how='inner', on='id')



result = pd.concat([result.drop(['old'], axis=1), result['old'].apply(pd.Series)], axis=1)

print(result.iloc[0])

result_build = pd.merge(build_df, result, how='inner', on='build_id')
print("shape before: ", result_build.shape)
result_build.set_index('build_id')
result_build = result_build[['build_id', 'log']]

result_build = result_build.groupby('build_id')['log'].apply(' '.join).reset_index()
print(result_build)
print("shape after: ", result_build.shape)

result = pd.merge(result, result_build, how='left', on='build_id')
result['newState'] = result['new'].apply(pd.Series)['state']

result['lenLogs']  = result['log_y'].str.len()

result['log'] = result['log_y']
result = result.drop(columns=['log_x', 'log_y', '_id_x', 'diff', 'analysis', 'logDiff', 'stage_id', 'number', 'state', 'started_at', 'finished_at', 'queue', 'allow_failure', 'tags', 'config'])

#v = TfidfVectorizer()
#x = v.fit_transform(result['log'])
#feature_names = v.getFeatureNames()

#x_train = pd.DataFrame.sparse.from_spmatrix(x, columns=feature_names)
print(result.head())
print(result.columns)
print(result.shape)
print(result.iloc[0])

#print(x_train.head())
#print(x_train.columns)
#print(x_train.shape)


elp_time = time.time() - start_time

print(elp_time)



result.to_pickle("/scratch2/2191051t/results.pkl")

elp_time = time.time() - start_time

print(elp_time)
