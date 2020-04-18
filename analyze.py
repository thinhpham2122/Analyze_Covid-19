import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def filter_feature(df):
    df.drop(columns=['FIPS', 'POVRATE'], inplace=True)
    return df


def filter_label(df):
    df = df[['fips_county', 'fips_state', 'state', 'type', '4/13/20']]
    # fips = (df[df.index % 2 == 0]['fips_state'] * 1000 + df[df.index % 2 == 0]['fips_county']).to_numpy()
    state = df[df.index % 2 == 0]['state'].to_numpy()
    cases = df[df.index % 2 == 0]['4/13/20'].to_numpy()
    deaths = df[df.index % 2 == 1]['4/13/20'].to_numpy()
    filtered_df = pd.DataFrame({'state': state, 'cases': cases, 'deaths': deaths})
    # print(filtered_df.head())
    return filtered_df


def get_features_labels(df_features, df_labels, max_population, max_median_income):
    total_populations = df_features['TOTALPOP']
    # normalize features
    df_features['POVEST'] = df_features['POVEST'].div(total_populations)
    df_features['GENDER_M'] = df_features['GENDER_M'].div(total_populations)
    df_features['GENDER_F'] = df_features['GENDER_F'].div(total_populations)
    df_features['AGE 0-19'] = df_features['AGE 0-19'].div(total_populations)
    df_features['AGE 20-49'] = df_features['AGE 20-49'].div(total_populations)
    df_features['AGE 50 ABOVE'] = df_features['AGE 50 ABOVE'].div(total_populations)
    df_features['TOTALPOP'] = df_features['TOTALPOP'].div(max_population)
    df_features['MEDINCOME'] = df_features['MEDINCOME'].div(max_median_income)
    # normalize labels
    df_labels['cases'] = df_labels['cases'].div(total_populations)
    df_labels['deaths'] = df_labels['deaths'].div(total_populations)
    return df_features.drop(columns=['STATE', 'CTYNAME']).to_numpy(), df_labels.drop(columns=['state']).to_numpy()


def make_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(output_size, input_dim=input_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    model.summary()
    return model


def main():
    # set options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # get data set
    raw_features = pd.read_csv('data/DATASET1_variables.csv')
    raw_labels = pd.read_csv('data/DataSet2_time_series.csv')

    # filter data set
    df_features, df_labels = filter_feature(raw_features), filter_label(raw_labels)

    # save important values
    max_population = df_features['TOTALPOP'].max()
    max_median_income = df_features['MEDINCOME'].max()

    # split GA and NY data set
    df_features_NY = df_features[df_features['STATE'] == 'NY']
    df_labels_NY = df_labels[df_labels['state'] == 'New York']
    df_features_GA = df_features[df_features['STATE'] == 'GA']
    df_labels_GA = df_labels[df_labels['state'] == 'Georgia']

    # get NY & GA features and labels
    features_NY, labels_NY = get_features_labels(df_features_NY, df_labels_NY, max_population, max_median_income)
    features_GA, labels_GA = get_features_labels(df_features_GA, df_labels_GA, max_population, max_median_income)

    # make and train model on NY features and labels
    model = make_model(len(features_NY[0]), len(labels_NY[0]))
    model.fit(features_NY, labels_NY, validation_data=(features_GA, labels_GA), epochs=500, batch_size=256, verbose=1)
    model.save('linear_regression')

    # predict on GA features
    predictions = np.maximum(model.predict(features_GA), np.zeros(labels_GA.shape))

    # make output data frame
    df_output = pd.DataFrame()
    df_output['state'] = df_features_GA['STATE']
    df_output['county'] = df_features_GA['CTYNAME']
    df_output['actual cases'] = df_labels_GA['cases']
    df_output['actual deaths'] = df_labels_GA['deaths']
    df_output['predicted cases'] = predictions[:, 0]
    df_output['predicted deaths'] = predictions[:, 1]
    df_output['cases different'] = df_output['predicted cases'] - df_output['actual cases']
    df_output['deaths different'] = df_output['predicted deaths'] - df_output['actual deaths']
    df_output.to_csv('predictions.csv')
    print(df_output)


if __name__ == '__main__':
    main()