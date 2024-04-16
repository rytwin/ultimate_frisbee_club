#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:45:03 2024

@author: ryandrost
"""

import numpy as np
import pandas as pd


df = pd.read_csv('data/cleaned_data.csv')
p = pd.read_csv('data/players.csv')[['player_id', 'display_name']]
games = pd.read_csv('data/games.csv')
tourney = pd.read_csv('data/tournaments.csv')


df['pass'] = np.where((df['thr_id'].notna()) & (df['rec_id'].notna()), 1, 0)
df['goal'] = np.where(((df['ast_att'] == 1) & (df['to'] == 0)) | (df['callahan'] == 1), 1, 0)
df['huck'] = np.where(df['gain_yd'] >= 35, 1, 0)
df['huck_cmp'] = np.where((df['huck'] == 1) & (df['to'] == 0), 1, 0)
df['dist_yd'] = np.where(df['to'] == 0, df['dist_yd'], 0)
df['gain_yd'] = np.where(df['to'] == 0, df['gain_yd'], 0)
df['horz_yd'] = np.where(df['to'] == 0, df['horz_yd'], 0)

t = df.groupby(['gm_id', 'thr_id']).agg({'dist_yd': 'sum', 'gain_yd': 'sum',
                                         'horz_yd': 'sum', 'thr_err': 'sum',
                                         'rec_err': 'sum', 'ast_att': 'sum',
                                         'hock_ast': 'sum', 'huck': 'sum',
                                         'stall': 'sum', 'pass': 'sum',
                                         'huck_cmp': 'sum', 'goal': 'sum'}).reset_index()
t.rename(columns = {'dist_yd': 'dist_thr', 'gain_yd': 'gain_thr',
                    'horz_yd': 'horz_thr', 'rec_err': 'acc_pass_dropped',
                    'huck': 'huck_thr', 'huck_cmp': 'huck_thr_cmp', 'goal': 'ast'}, inplace=True)

r = df.groupby(['gm_id', 'rec_id']).agg({'dist_yd': 'sum', 'gain_yd': 'sum',
                                         'horz_yd': 'sum', 'rec_err': 'sum',
                                         'ast_att': 'sum', 'blk': 'sum',
                                         'callahan': 'sum', 'huck': 'sum',
                                         'drop_pull': 'sum', 'pass': 'sum',
                                         'thr_err': 'sum', 'huck_cmp': 'sum',
                                         'goal': 'sum'}).reset_index()
r.rename(columns = {'dist_yd': 'dist_rec', 'gain_yd': 'gain_rec', 'thr_err': 'uncatch_rec',
                    'horz_yd': 'horz_rec', 'rec_err': 'drop', 'pass': 'int_rec',
                    'huck': 'huck_rec', 'ast_att': 'goal_att', 'huck_cmp': 'huck_rec_cmp'}, inplace=True)

dg = t.merge(r, how = 'outer', left_on = ['gm_id', 'thr_id'], right_on = ['gm_id', 'rec_id'])
dg = dg.fillna(0)
dg['ast'] = dg['ast']
dg['score'] = dg['goal'] + dg['ast']
dg['to'] = dg['thr_err'] + dg['stall'] + dg['drop']
dg['compl'] = dg['pass'] - dg['acc_pass_dropped'] - dg['thr_err']
dg['acc_pass'] = dg['pass'] - dg['thr_err']
dg['touch'] = dg['pass'] + dg['goal'] + dg['drop'] + dg['stall']
dg['rec'] = dg['int_rec'] - dg['uncatch_rec']
dg['catch'] = dg['rec'] - dg['drop']
dg['thr_id'] = np.where(dg['thr_id'] == 0, dg['rec_id'], dg['thr_id'])
dg['rec_id'] = np.where(dg['rec_id'] == 0, dg['thr_id'], dg['rec_id'])
dg['tot_yd'] = dg['gain_rec'] + dg['gain_thr']
dg.rename(columns = {'thr_id': 'player_id'}, inplace=True)



team_tot = df.groupby(['gm_id', 'point', 'poss_id']).agg({'goal': 'sum', 'callahan': 'sum', 'pass': 'sum',
                                                          'rec_err': 'sum', 'stall': 'sum'}).reset_index()
team_tot['tm_touch'] = team_tot['pass'] + team_tot['rec_err'] + team_tot['stall'] + team_tot['goal']
team_tot = team_tot[['gm_id', 'point', 'poss_id', 'tm_touch']]

poss = df.groupby(['gm_id', 'point', 'poss_id']).max(numeric_only = True).reset_index()[['gm_id', 'point', 'poss_id', 'o_pt',
                                                                                         'scored_on_poss', 'scored_on_pt']]
poss = poss.merge(team_tot, how = 'left', on = ['gm_id', 'point', 'poss_id'])

tm_conv = poss.groupby('gm_id').agg({'poss_id': 'count', 'scored_on_poss': 'sum'}).reset_index()

pp = pd.read_csv('data/points_played.csv').merge(poss, how = 'left', on=['gm_id', 'point', 'poss_id'])
pp['o_pt'].fillna(0, inplace = True)
pp['scored_on_pt'].fillna(0, inplace = True)
pp['scored_on_poss'].fillna(0, inplace = True)
players = pd.read_csv('data/players.csv')[['Player', 'player_id']]
pp = pp.merge(players, how = 'left', left_on = 'player', right_on = 'Player')
by_poss = pp.groupby(['player', 'player_id', 'gm_id']).agg({'poss_id': 'count', 'scored_on_poss': 'sum', 'tm_touch': 'sum'}).reset_index()
by_poss.rename(columns = {'poss_id': 'poss', 'scored_on_poss': 'conv'}, inplace = True)
by_oposs = pp[pp['o_pt'] == 1].groupby(['player', 'player_id', 'gm_id']).agg({'poss_id': 'count', 'scored_on_poss': 'sum'}).reset_index()
by_oposs.rename(columns = {'poss_id': 'o_poss', 'scored_on_poss': 'o_conv'}, inplace = True)
by_dposs = pp[pp['o_pt'] == 0].groupby(['player', 'player_id', 'gm_id']).agg({'poss_id': 'count', 'scored_on_poss': 'sum'}).reset_index()
by_dposs.rename(columns = {'poss_id': 'd_poss', 'scored_on_poss': 'd_conv'}, inplace = True)


pts = poss.drop_duplicates(subset = ['gm_id', 'point']).reset_index(drop=True)[['gm_id', 'point', 'o_pt', 'scored_on_pt']]
all_pts = pd.read_csv('data/points_played.csv')[['player', 'point', 'gm_id']].drop_duplicates()
pts_merge = all_pts.merge(pts, how = 'left', on=['gm_id', 'point']).merge(players, how = 'left', left_on = 'player', right_on = 'Player')
pts_merge['o_pt'].fillna(0, inplace = True)
pts_merge['scored_on_pt'].fillna(0, inplace = True)

tm_pts = pd.read_csv('data/points_played.csv')[['point', 'gm_id']].drop_duplicates()
tm_pts = tm_pts.merge(pts, how = 'left', on=['gm_id', 'point']).fillna(0)
tm_pts['tm_o_scores'] = np.where((tm_pts['o_pt'] == 1) & (tm_pts['scored_on_pt'] == 1), 1, 0)
tm_pts['tm_d_scores'] = np.where((tm_pts['o_pt'] == 0) & (tm_pts['scored_on_pt'] == 1), 1, 0)
tm_pts = tm_pts.groupby('gm_id').agg({'point': 'count', 'o_pt': 'sum', 'tm_o_scores': 'sum', 'tm_d_scores': 'sum'}).reset_index()
tm_pts.rename(columns = {'point': 'tm_pts', 'o_pt': 'tm_o_pt'}, inplace = True)
tm_pts['tm_d_pt'] = tm_pts['tm_pts'] - tm_pts['tm_o_pt']

by_pts = pts_merge.groupby(['player', 'player_id', 'gm_id']).agg({'point': 'count', 'scored_on_pt': 'sum'}).reset_index()
by_pts.rename(columns = {'point': 'pts', 'scored_on_pt': 'tm_score'}, inplace = True)
by_opt = pts_merge[pts_merge['o_pt'] == 1].groupby(['player', 'player_id', 'gm_id']).agg({'point': 'count', 'scored_on_pt': 'sum'}).reset_index()
by_opt.rename(columns = {'point': 'o_pts', 'scored_on_pt': 'o_score'}, inplace = True)
by_dpt = pts_merge[pts_merge['o_pt'] == 0].groupby(['player', 'player_id', 'gm_id']).agg({'point': 'count', 'scored_on_pt': 'sum'}).reset_index()
by_dpt.rename(columns = {'point': 'd_pts', 'scored_on_pt': 'd_score'}, inplace = True)

by_poss = by_poss.drop('player', axis=1)
by_oposs = by_oposs.drop('player', axis=1)
by_dposs = by_dposs.drop('player', axis=1)
by_pts = by_pts.drop('player', axis=1)
by_opt = by_opt.drop('player', axis=1)
by_dpt = by_dpt.drop('player', axis=1)




data = dg.merge(by_poss, how = 'outer', on = ['player_id', 'gm_id']).merge(by_oposs, how = 'outer', on = ['player_id', 'gm_id']).merge(by_dposs, how = 'outer', on = ['player_id', 'gm_id'])
data = data.merge(by_pts, how = 'outer', on = ['player_id', 'gm_id']).merge(by_opt, how = 'outer', on = ['player_id', 'gm_id']).merge(by_dpt, how = 'outer', on = ['player_id', 'gm_id'])
data = data.merge(games, how = 'left', on = 'gm_id').merge(tourney, how = 'left', on = 'trn_id')
data['Wind speed (mph)'] = data['Wind speed (km/h)'] * 0.621371


data = data.merge(p, how = 'left', on = 'player_id')
data = data.merge(tm_pts, how = 'left', on = 'gm_id')
data = data.fillna(0)
data['opp_o_poss'] = data['d_pts'] + data['d_poss'] - data['d_score'] + data['o_poss'] - data['o_score']
data['o_earned_poss'] = data['o_poss'] - data['o_pts']
data['tm_opp_o_poss'] = data['tm_d_pt'] + data['Defense possessions'] - data['tm_d_scores'] + data['Offense possessions'] - data['tm_o_scores']
data['tm_o_earned_poss'] = data['Offense possessions'] - data['tm_o_pt']




data.to_csv('data/tableau_data.csv', index=False)


