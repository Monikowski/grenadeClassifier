import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

'''
'''
def make_dummies(categories, column):
    values = np.array(categories)
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit(values).transform(column)
    onehot_encoder = OneHotEncoder(sparse=False)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    df = pd.DataFrame(onehot_encoded, columns=categories)
    return df

def make_all_dummies(cat_col_val, source):
    all_dummies = None
    for category in cat_col_val:
        if all_dummies is None:
            all_dummies = make_dummies(cat_col_val[category], source[category])
        else:
            all_dummies = pd.concat([all_dummies, make_dummies(cat_col_val[category], source[category])], axis=1)
    return all_dummies


parser = argparse.ArgumentParser(description='Give path to file with grenade data')
parser.add_argument('path', metavar='path', type=str,
                    help='path to file')
args = parser.parse_args()
path = args.path

grenades = pd.read_csv(path)
grenades = grenades.loc[:, ['team',
                            'detonation_raw_x', 'detonation_raw_y', 'detonation_raw_z',
                            'throw_from_raw_x', 'throw_from_raw_y', 'throw_from_raw_z',
                            'throw_tick', 'detonation_tick',
                            'TYPE', 'map_name']]
grenades['throw_dist'] = np.sqrt((grenades['throw_from_raw_x'] - grenades['detonation_raw_x'])**2 +
                                   (grenades['throw_from_raw_y'] - grenades['detonation_raw_y'])**2 +
                                   (grenades['throw_from_raw_z'] - grenades['detonation_raw_z'])**2)
grenades = grenades[['detonation_raw_x', 'detonation_raw_y', 'detonation_raw_z',
                    'team', 'TYPE', 'map_name', 'throw_tick', 'throw_dist']]
print(grenades)

category_cols_vals = {}
category_cols_vals["team"] = ["CT", "T"]
category_cols_vals["TYPE"] = ["flashbang", "molotov", "smoke"]
category_cols_vals["map_name"] = ["de_inferno", "de_mirage"]

dummies = make_all_dummies(category_cols_vals, grenades)

print(dummies)

'''
team_dummies = make_dummies(teams, team_col)
team_df = pd.DataFrame(team_dummies, columns=['team_CT', 'team_T'])
print(team_df)

type_col = grenades["TYPE"]
type_dummies = make_dummies(types, type_col)
type_df = pd.DataFrame(type_dummies, columns=['TYPE_flashbang', 'TYPE_molotov', 'TYPE_smoke'])
print(type_df)

map_col = grenades["map_name"]
map_dummies = make_dummies(maps, map_col)
map_df = pd.DataFrame(map_dummies, columns=['map_name_de_inferno', 'map_name_de_mirage'])
print(map_df)
'''

