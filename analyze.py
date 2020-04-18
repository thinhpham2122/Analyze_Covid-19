import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow import set_random_seed
set_random_seed(5)


def filter_raw_feature(df):
    df.drop(columns=['FIPS', 'POVRATE'], inplace=True)
    return df


def filter_raw_label(df):
    df = df[['fips_county', 'fips_state', 'state', 'type', '4/13/20']]
    state = df[df.index % 2 == 0]['state'].to_numpy()
    cases = df[df.index % 2 == 0]['4/13/20'].to_numpy()
    deaths = df[df.index % 2 == 1]['4/13/20'].to_numpy()
    filtered_df = pd.DataFrame({'state': state, 'cases': cases, 'deaths': deaths})
    return filtered_df


def filter_combine_df(raw_feature, raw_labels):
    # filter features
    filtered_feature = raw_feature.drop(columns=['FIPS', 'POVRATE'])

    # filter labels
    cases = raw_labels[raw_labels.index % 2 == 0]['4/13/20'].to_numpy()
    deaths = raw_labels[raw_labels.index % 2 == 1]['4/13/20'].to_numpy()
    filtered_label = pd.DataFrame({'cases': cases, 'deaths': deaths})

    # combine 2 data sets
    df = pd.concat([filtered_feature, filtered_label], axis=1, sort=False)
    return df


def get_features_labels(df, max_population, max_median_income):
    # normalize features
    df_features = pd.DataFrame()
    df_features['POVEST'] = df['POVEST'].div(df['TOTALPOP'])
    df_features['GENDER_M'] = df['GENDER_M'].div(df['TOTALPOP'])
    df_features['GENDER_F'] = df['GENDER_F'].div(df['TOTALPOP'])
    df_features['AGE 0-19'] = df['AGE 0-19'].div(df['TOTALPOP'])
    df_features['AGE 20-49'] = df['AGE 20-49'].div(df['TOTALPOP'])
    df_features['AGE 50 ABOVE'] = df['AGE 50 ABOVE'].div(df['TOTALPOP'])
    df_features['TOTALPOP'] = df['TOTALPOP'].div(max_population)
    df_features['MEDINCOME'] = df['MEDINCOME'].div(max_median_income)

    # normalize labels
    df_labels = pd.DataFrame()
    df_labels['cases'] = df['cases'].div(df['TOTALPOP']) * 100
    df_labels['deaths'] = df['deaths'].div(df['TOTALPOP']) * 100
    return df_features.to_numpy(), df_labels.to_numpy()


def make_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(output_size, input_dim=input_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    model.summary()
    return model


def plot_bar(df):
    labels = df['county'].to_numpy()
    actual_cases = df['actual cases'].to_numpy()
    predicted_cases = df['predicted cases'].to_numpy()
    actual_deaths = df['actual deaths'].to_numpy()
    predicted_deaths = df['predicted deaths'].to_numpy()
    x = np.arange(len(labels))  # the label locations
    width = 1/5  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width * 1.5, actual_cases, width, label='actual cases')
    rects2 = ax.bar(x - width / 2, actual_deaths, width, label='actual deaths')
    rects3 = ax.bar(x + width / 2, predicted_cases, width, label='predicted cases')
    rects4 = ax.bar(x + width * 1.5, predicted_deaths, width, label='predicted deaths')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percent of population')
    ax.set_title('Georgia cases and deaths')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def main():
    # set options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # get data set
    raw_features = pd.read_csv('data/DATASET1_variables.csv')
    raw_labels = pd.read_csv('data/DataSet2_time_series.csv')

    # filter data set
    filtered_df = filter_combine_df(raw_features, raw_labels)

    # save important values
    max_population = filtered_df['TOTALPOP'].max()
    max_median_income = filtered_df['MEDINCOME'].max()

    # split GA and NY data set
    df_NY = filtered_df[filtered_df['STATE'] == 'NY']
    df_GA = filtered_df[filtered_df['STATE'] == 'GA']

    # get 20 best counties to train on
    df_NY = df_NY.sort_values(by=['cases'], ascending=False).head(20)

    # get NY & GA features and labels
    features_NY, labels_NY = get_features_labels(df_NY, max_population, max_median_income)
    features_GA, labels_GA = get_features_labels(df_GA, max_population, max_median_income)

    # make and train model on NY features and labels
    model = make_model(len(features_NY[0]), len(labels_NY[0]))
    model.fit(features_NY, labels_NY, validation_data=(features_GA, labels_GA), epochs=100, batch_size=256, verbose=0)
    model.save('keras_model_linear_reg')

    # predict on GA features
    predictions = np.maximum(model.predict(features_GA), np.zeros(labels_GA.shape))

    # make output data frame
    df_output = pd.DataFrame()
    df_output['state'] = df_GA['STATE']
    df_output['county'] = df_GA['CTYNAME']
    df_output['actual cases'] = labels_GA[:, 0]
    df_output['actual deaths'] = labels_GA[:, 1]
    df_output['predicted cases'] = predictions[:, 0]
    df_output['predicted deaths'] = predictions[:, 1]

    # plot first 10 and save output
    plot_bar(df_output.sort_values(by=['actual cases'], ascending=False).head(10))
    print(df_output)
    df_output.to_csv('predictions.csv')


if __name__ == '__main__':
    main()