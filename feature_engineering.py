#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 14:03:07 2024

@author: ryandrost
"""

import numpy as np
import pandas as pd


df = pd.read_csv('data/cleaned_data.csv').dropna(subset=['thr_id', 'rec_id'])

# create new features
df['num_passes'] = df.groupby('poss_id').cumcount()
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['decayed_rtg'] = np.where(df['month'] == 6, 0.7, np.where(df['month'] == 7, 0.8, np.where(df['month'] == 8, 0.9, 1))) * df['opp_rtg']
df['upwind_dist'] = np.where(df['upwind'] == 1, df['dist_from_ez'], 0)
df['downwind_dist'] = np.where(df['downwind'] == 1, df['dist_from_ez'], 0)
df['max_swing'] = df.groupby('poss_id').cummax(numeric_only=True)['horz_yd']

# select columns
df = df[['action_id', 'gm_id', 'point', 'poss', 'poss_id', 'thr_id', 'rec_id', 'num_passes',
         'dist_from_ez', 'dist_from_middle', 'dist_from_mid_ez',
         'from_pull', 'o_pt', 'upwind', 'downwind', 'wind_speed', 
         'opp_rank', 'opp_rtg', 'decayed_rtg', 'bracket', 'consolation', 'other',
         'postseason', 'tct', 'downwind_dist', 'upwind_dist', 'scored_on_poss', 'scored_on_pt']]

# add on/off binary player variables
lines = pd.read_csv('data/points_played.csv')
players = pd.read_csv('data/players.csv')
lines = pd.merge(lines, players, how='left', left_on=['player', 'year'], right_on=['Player', 'Season'])
lines.drop(['player', 'Player', 'Season', 'year'], axis=1, inplace=True)
lines['on'] = 1
lines_wide = lines.pivot(index=['gm_id', 'point', 'poss_id'], columns='player_id', values='on').fillna(0).reset_index()
for c in lines_wide.columns:
    if type(c) == int:
        lines_wide.rename(columns = {c: f'player_{c}'}, inplace=True)
lines_wide.drop(['gm_id', 'point'], axis=1, inplace=True)

df = pd.merge(df, lines_wide, how='left', on=['poss_id']).drop(['poss_id'], axis=1)

# use this to check that there are 7 players on the field
# except for instances of injury mid-possession, then there may be more than 7
player_columns = df.filter(regex=r'^player_\d+$')
players_on_field_check = player_columns.sum(axis=1)



# save file
df.to_csv('data/model_data.csv', index=False)
print('Updated data for model saved as model_data.csv')
