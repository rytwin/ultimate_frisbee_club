#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:33:51 2024

@author: ryandrost
"""

import numpy as np
import pandas as pd


### IMPORTANT (manual updates):
# when adding data from new games, you need to do the following:
# create txt file with game_id in folder for that game
# add game folder name (ex: 'Wild Card 2023-06-24_09-00-00') with game id to game_list.csv
# add any stalls/blocks/callahans/dropped pulls to the passes.csv file
# add trn_id, game_type, opp_region, opp_rank, opp_rtg columns, gm_id, & folder to games.csv file
# add postseason and tct variables to tournaments.csv file
# add wind direction to points.csv file
# modify passes.csv (before running) and points_played.csv for injury substitutions (after running)
# for new seasons, players.csv file will need to be updated with additional player id mapping

# known issue: injury subs are only recorded if both players played an offensive possession on that point
#               otherwise, only the player who played an offensive possession on that point is recorded
#               or, if neither did, the player who started the point is recorded
#               so yes, it's possible if someone gets a block and then leaves the game,
#               they can have a block on a point where it says they didn't play and weren't on the field


# modify these first 3 lines to add data (then run the file)
# when first_save is True, it will overwrite any existing file; when false, it will add to that file
# when first_save is True, it will also pull the file names automatically from game_list, so
#   the game_ids variable directly below is irrelevant
# when save_lines is True, it will also save points played for every player
first_save = False
save_lines = True
game_ids = [] #if first_save is False, enter the game_ids to be added here


############################# no need to edit below here! ######################
if not first_save:
    data = pd.read_csv('data/cleaned_data.csv')
    start_id = np.max(data['action_id']) + 1
    file_names = pd.read_csv('data/games/game_list.csv')
    file_names = list(file_names[file_names['gm_id'].isin(game_ids)]['folder'])
else:
    start_id = 1
    file_names = list(pd.read_csv('data/games/game_list.csv')['folder'])

passes_list = []
points_list = []
for f in file_names:
    df1 = pd.read_csv(f'data/games/{f}/Passes vs. {f}.csv')
    df1['gm_id'] = int(np.loadtxt(f'data/games/{f}/game_id.txt'))
    df2 = pd.read_csv(f'data/games/{f}/Points vs. {f}.csv')
    df2['gm_id'] = int(np.loadtxt(f'data/games/{f}/game_id.txt'))
    passes_list.append(df1)
    points_list.append(df2)

passes = pd.concat(passes_list, ignore_index=True)
points = pd.concat(points_list, ignore_index=True)


df = pd.merge(passes, points, how='left', on=['Point', 'gm_id'])
df['action_id'] = range(start_id, start_id + len(df))
df['poss_id'] = df['gm_id'].astype(str) + '.' + df['Point'].astype(str) + '.' + df['Possession'].astype(str)
df['pt_id'] = df['gm_id'].astype(str) + '.' + df['Point'].astype(str)
scored_on_poss = df.groupby('poss_id')['Assist?'].max()
scored_on_pt = df.groupby('pt_id')['Assist?'].max()

df['scored_on_poss'] = df['poss_id'].map(scored_on_poss)
df['scored_on_pt'] = df['pt_id'].map(scored_on_pt)

games = pd.read_csv('data/games.csv')
game_ids = pd.read_csv('data/games/game_list.csv')
games['Date'] = games['Date'] + ' ' + games['Time']
games['date'] = pd.to_datetime(games['Date'], format = '%m/%d/%y %H:%M', errors='coerce')
games.sort_values(by = 'date', inplace=True)
games['year'] = games['date'].dt.year

df = pd.merge(df, games, how='left', on='gm_id')
df.rename(columns = {'Start X (0 -> 1 = left sideline -> right sideline)': 'start_x',
                     'Start Y (0 -> 1 = back of opponent endzone -> back of own endzone)': 'start_y',
                     'End X (0 -> 1 = left sideline -> right sideline)': 'end_x',
                     'End Y (0 -> 1 = back of opponent endzone -> back of own endzone)': 'end_y',
                     'Started on offense?': 'o_pt',
                     'Wind speed (km/h)': 'wind_speed',
                     'Assist?': 'goal', 'Throw to endzone?': 'ast_att',
                     'Turnover?': 'to', 'Thrower error?': 'thr_err',
                     'Receiver error?': 'rec_err', 'Secondary assist?': 'hock_ast',
                     'Swing?': 'swing', 'Dump?': 'dump',
                     'From sideline?': 'from_side', 'To sideline?': 'to_side',
                     'Point': 'point', 'Possession': 'poss'}, inplace=True)

meter_yd_conversion = 1.09361
df['dist_yd'] = df['Distance (m)'] * meter_yd_conversion
df['gain_yd'] = df['Forward distance (m)'] * meter_yd_conversion
df['lr_yd'] = df['Left-to-right distance (m)'] * meter_yd_conversion
df['horz_yd'] = abs(df['lr_yd'])
df['huck'] = np.where(df['gain_yd'] > 40, 1, 0)

# add player ids from players.csv
players = pd.read_csv('data/players.csv')
df = pd.merge(df, players, how='left', left_on=['Thrower', 'year'], right_on=['Player', 'Season'])
df.rename(columns = {'player_id': 'thr_id', 'matchup': 'thr_match'}, inplace=True)
df = pd.merge(df, players, how='left', left_on=['Receiver', 'year'], right_on=['Player', 'Season'])
df.rename(columns = {'player_id': 'rec_id', 'matchup': 'rec_match'}, inplace=True)

# merge with tourney info
tourney = pd.read_csv('data/tournaments.csv')
df = pd.merge(df, tourney, how='left', on='trn_id')

# add blocks, callahans, dropped pulls
df['blk'] = np.where((df['Thrower'].isna()) & (df['to'] == 0), 1, 0)
df['callahan'] = np.where((df['blk'] == 1) & (df['goal'] == 1), 1, 0)
df['stall'] = np.where(df['Receiver'].isna(), 1, 0)
df['drop_pull'] = np.where((df['Thrower'].isna()) & (df['rec_err'] == 1), 1, 0)

df = df[['action_id', 'year', 'date', 'trn_id', 'gm_id', 'poss_id', 'point', 'poss',
         'thr_id', 'rec_id', 'thr_match', 'rec_match', 'start_x', 'start_y', 'end_x', 'end_y',
         'dist_yd', 'gain_yd', 'horz_yd', 'o_pt', 'wind_speed', 'wind_direction', 'game_type',
         'opp_region', 'opp_rank', 'opp_rtg', 'postseason', 'tct', 'to', 'thr_err', 'rec_err',
         'ast_att', 'hock_ast', 'huck', 'swing', 'dump', 'from_side', 'to_side', 'blk', 'callahan',
         'stall', 'drop_pull', 'scored_on_poss', 'scored_on_pt']]

df['dist_from_middle'] = abs(df['start_x'] - 0.5) * 40
df['dist_from_ez'] = (df['start_y'] * 110) - 20
df['dist_from_ez'] = df['dist_from_ez'].apply(lambda x: 0.5 if x <= 0 else x)
df['dist_from_mid_ez'] = np.sqrt(df['dist_from_middle']**2 + df['dist_from_ez']**2)
df['bracket'] = df['game_type'].apply(lambda x: 1 if x == 'bracket' else 0)
df['consolation'] = df['game_type'].apply(lambda x: 1 if x == 'consolation' else 0)
df['other'] = df['game_type'].apply(lambda x: 0 if x in ['bracket', 'consolation'] else 1)
df['from_pull'] = df.apply(lambda row: 1 if (row['o_pt'] == 1 and row['poss'] == 1) else 0, axis=1)
df['upwind'] = np.where(df['wind_direction'] == 'u', 1, 0)
df['downwind'] = np.where(df['wind_direction'] == 'd', 1, 0)

# create points played file
stats_list = []
for f in file_names:
    df3 = pd.read_csv(f'data/games/{f}/Player Stats vs. {f}.csv')
    df3 = df3[['Player', 'Points played']]
    df3['gm_id'] = int(np.loadtxt(f'data/games/{f}/game_id.txt'))
    df3['Points played'] = df3['Points played'].str.split(',')
    df3 = df3.explode('Points played').reset_index(drop=True)
    df3.rename(columns = {'Player': 'player', 'Points played': 'point'}, inplace=True)
    df3['point'] = df3['point'].astype(int)
    stats_list.append(df3)

stats = pd.concat(stats_list, ignore_index=True)
stats = pd.merge(stats, games, how='left', on='gm_id')
stats = stats[['player', 'point', 'gm_id', 'year']]

possessions = df[['gm_id', 'point', 'poss_id']]
stats = pd.merge(stats, possessions, how='left', on=['gm_id', 'point']).drop_duplicates().reset_index(drop=True)
stats['injury'] = 0

# save files
if first_save:
    if save_lines:
        stats.to_csv('data/points_played.csv', index=False)
        print('New file created: points_played.csv')
    df.to_csv('data/cleaned_data.csv', index=False)
    print('New file created: cleaned_data.csv')
else:
    if save_lines:
        pp = pd.read_csv('data/points_played.csv')
        pp = pd.concat([pp, stats])
        pp.to_csv('data/points_played.csv', index=False)
        print('New data added to points_played.csv')
    data = pd.concat([data, df], ignore_index=True)
    data.to_csv('data/cleaned_data.csv', index=False)
    print('New data added to cleaned_data.csv')
    
    
