# ultimate_frisbee_club

This is to be used in conjunction with the Statto app. After inputting the stats for a game(s) in the app, export the game files and save them in the `data/games/XXXXXX` directory, where XXXXXX is the opponent name, date, and time included in the automatically exported files (e.g.: “Hybrid 2023-10-21_10-00-00”).

1. There are some additional manual updates necessary:

-Add the directory name and game id (just keep going in sequential order for ID) to the `game_list.csv` file in `data/games/`

-Create a txt file with just the `game_id` in the `data/games/XXXXXX` directory (titled `game_id`)

-Copy the game information into  `games.csv` in `data/` folder. Add the info for `trn_id`, `game_type`, `opp_region`, `opp_rank`, `opp_rtg`, `gm_id`, and `folder`.

-Enter `postseason` and `tct` variables in `data/tournaments.csv`

-Add any stalls (`Stall Outs Against.csv`), blocks and callahans (`Defensive Blocks.csv`), and dropped pulls (manual) for each game to the corresponding `Passes vs. XXXXXX.csv` file. You can use the `Created` field to insert in the correct order. Copy the `Created`, `Point`, `Player`, and `Location X` and `Location Y` fields, and enter 0 for everything else. The `Location X` and `Location Y` fields should be entered as both the `Start` and `End` location for these events. The `Player` should be entered as the `Receiver` for blocks, callahans, and dropped pulls, with the `Thrower` left blank; for stalls, do the opposite. Lastly, for callahans enter “1” in the `Assist?` column, for dropped pulls enter “1” in the `Turnover?` Column, and for stalls enter “1” in both the `Turnover?` and `Thrower error?` columns.

-Add `wind_direction` to the `data/games/XXXXXX/Points vs. XXXXXX.csv file`. “u” for upwind, “d” for downwind, and blank for neither.

-Lastly, for new seasons you will need to update the `data/players.csv` file with the new rosters.


Known issue: Injury subs aren’t tracked completely accurately. Injury subs are only recorded if both players played an offensive possession on that point--otherwise, only the player who played an offensive possession on that point is recorded or, if neither did, the player who started the point is recorded.

2. Run `clean_imported_statto_data.py`. You will need to update 2-3 arguments within the file. This cleans the data and gets it in a more usable format with better column names. It is saved as `data/cleaned_data.csv`.
   
3. If you want to update Tableau dashboards, run `tableau_prep.py`. This takes `data/cleaned_data.csv` and organizes the data by player and game for easy analysis in Tableau and is saved as `data/tableau_data.csv`.
   
4. Run `feature_engineering.py`. It takes `data/cleaned_data.csv` and creates features for modeling in `data/model_data.csv` (the biggest one being dummy variables for whether each player was on the field or not). You may want to edit this to create more features.

5. Run `train.py`. This is where you train the model and work on feature selection. Once you have a final model, you can save all of the model details with this script.
