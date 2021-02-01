import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib
from tensorflow import keras
from csv import writer
from csv import reader
from tempfile import NamedTemporaryFile
import shutil


def _make_dummies(categories, column):
    """
    Creates onehot encoded data-frame using all possible values of a category column
    :param categories: list of possible category values
    :param column: column from source dataframe
    :return: pandas dataframe that holds onehot encoded columns
    """
    values = np.array(categories)
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit(values).transform(column)
    onehot_encoder = OneHotEncoder(sparse=False)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    df = pd.DataFrame(onehot_encoded, columns=categories)
    return df


def _make_all_dummies(cat_col_val, source):
    """
    Uses a dictionary cat_col_val of key: name of the column in source dataframe -> value: list of possible values
    in the category column to create onehot encoded categorical columns
    :param cat_col_val:  dictionary holding information about column names and all possible values
    :param source: source dataframe
    :return: dataframe with onehot encoded categorical variables
    """
    all_dummies = None
    for category in cat_col_val:
        if all_dummies is None:
            all_dummies = _make_dummies(cat_col_val[category], source[category])
        else:
            all_dummies = pd.concat([all_dummies, _make_dummies(cat_col_val[category], source[category])], axis=1)
    return all_dummies


if __name__ == "__main__":
    # setting the options for command line calls
    parser = argparse.ArgumentParser(description='Give path to file with grenade data')
    parser.add_argument('path', metavar='path', type=str,
                        help='path to file')
    args = parser.parse_args()

    # path to our file, if file is saved in script dir then file name
    path = args.path

    # loading the input data from csv
    grenades = pd.read_csv(path)

    # making sure we only take the columns we need
    grenades = grenades.loc[:, ['team',
                                'detonation_raw_x', 'detonation_raw_y', 'detonation_raw_z',
                                'throw_from_raw_x', 'throw_from_raw_y', 'throw_from_raw_z',
                                'throw_tick', 'detonation_tick',
                                'TYPE', 'map_name']]

    # calculating throw distance for each grenade
    grenades['throw_dist'] = np.sqrt((grenades['throw_from_raw_x'] - grenades['detonation_raw_x'])**2 +
                                       (grenades['throw_from_raw_y'] - grenades['detonation_raw_y'])**2 +
                                       (grenades['throw_from_raw_z'] - grenades['detonation_raw_z'])**2)
    grenades = grenades[['detonation_raw_x', 'detonation_raw_y', 'detonation_raw_z',
                        'team', 'TYPE', 'map_name', 'throw_tick', 'throw_dist']]

    # creating a dictionary that stores all possible categories for each categorical variable
    category_cols_vals = {}
    category_cols_vals["team"] = ["CT", "T"]
    category_cols_vals["TYPE"] = ["flashbang", "molotov", "smoke"]
    category_cols_vals["map_name"] = ["de_inferno", "de_mirage"]

    # creating onehot categorical columns
    dummies = _make_all_dummies(category_cols_vals, grenades)

    # taking only the numerical values from the dataframe
    grenades_num = grenades[['detonation_raw_x', 'detonation_raw_y', 'detonation_raw_z', 'throw_tick', 'throw_dist']]

    # combining onehot columns with numerical values
    X = pd.concat([dummies, grenades_num], axis=1)

    # changing the distribution of throw_tick so that it is more normal, this is for better standard scaler performance
    # later
    X["throw_tick_norm"] = np.log(X["throw_tick"] + 1)
    X.drop("throw_tick", 1, inplace=True)

    # loading the scaler which learned on entirety of train data
    scaler = joblib.load('scaler.gz')

    # creating the scaled numerical variables
    X_scaled = X.copy()
    X_scaled[["detonation_raw_x", "detonation_raw_y", "detonation_raw_z",
              "throw_dist", "throw_tick_norm"]] = scaler.transform(X_scaled[["detonation_raw_x",
                                                                                 "detonation_raw_y", "detonation_raw_z",
                                                                                 "throw_dist", "throw_tick_norm"]])

    # loading the saved model and making predictions
    model = keras.models.load_model('.')
    predictions = model.predict_classes(X_scaled)

    # changing the predictions format to list so that it can be mapped
    predictions = predictions.tolist()

    # changing prediction format from digits to TRUE or FALSE for 1 and 0 respectively
    result = [*map(lambda x: 'TRUE' if x[0] == 1 else 'FALSE', predictions)]

    # creating a temporary file to which we will write our grenade data and predictions for each grenade
    temp_file = NamedTemporaryFile('w+t', newline='', delete=False)

    # opening the input data file to carry over and append the prediciton to each grenade
    with open(path, 'r') as read_obj, temp_file:
        csv_writer = writer(temp_file, delimiter=',')
        csv_reader = reader(read_obj, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # appending RESULT to the first (header) row
                row.append("RESULT")
            else:
                # appending the prediction to each grenade's row
                row.append(list(result)[line_count - 1])
            line_count += 1
            csv_writer.writerow(row)
    shutil.move(temp_file.name, path)
