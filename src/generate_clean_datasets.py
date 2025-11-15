

# This file contains all the code for creating a clean dataset for the /data/home_insulin.csv
# There are a bunch of problems with it, which are commented as the code fixes them.

from datetime import datetime
import numpy as np
import pandas as pd

# This method adds a new column for the average blood sugar level based on the month.
# My thinking is this would be useful so the model would see the target glucose level changing slightly.
# If we set this value to the target clucose, rather than the model only ever seeing the same number.
def create_target_gc_df(df: pd.DataFrame) -> pd.DataFrame:

    df['year_month'] = df['dateTime'].dt.to_period('M')  # e.g., 2020-08

    monthly_avg = df.groupby('year_month')['bloodGlucose'].transform('mean')

    df['averageMonthlyBloodGlucose'] = monthly_avg

    df.drop(columns=['year_month'], inplace=True)

    return df

# This method creates a dataset for how I imagine this would be used for LTSM or other time series models.
def create_time_series(df: pd.DataFrame)-> pd.DataFrame:

    df['date'] = df['dateTime'].dt.date

    max_events = df.groupby('date').size().max()

    rows = []

    for date, group in df.groupby('date'):
        row = {'date': date}
        for i, (_, event) in enumerate(group.iterrows()):
            row[f'event_{i+1}'] = event['event']
            row[f'netCarbs_{i+1}'] = event['netCarbs']
            row[f'bloodGlucose_{i+1}'] = event['bloodGlucose']
            row[f'insulinTaken_{i+1}'] = event['insulinTaken']

        for j in range(len(group), max_events):
            row[f'event_{j+1}'] = ''
            row[f'netCarbs_{j+1}'] = 0
            row[f'bloodGlucose_{j+1}'] = 0
            row[f'insulinTaken_{j+1}'] = 0
        rows.append(row)

    df_series = pd.DataFrame(rows)

    df.drop(columns=['date'], inplace=True)

    return df_series


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # day,eventTime,event,bloodGlucose,netCarbs,insulinToCarbRatio,actualInsulinAmount,recomendedInsulinAmount,insulinSensitivityFactor,bloodGlucoseTarget
    df = df.rename(columns={
        "day": "date",
        })

    ### Fix the event column which has a bunch of stuff we don't want to use
    # events = set(df['event'])
    # {
    #     'Workout', 'diner', 'breakfast', 'test', 'lantus', 'michelle', 
    #     'lunb', 'workout', 'es', 'est', 'tst', 'Test', 'ewo', 'Dinner', 
    #     'lunch', 'mich shake', 'unch', 'dinner', nan, 'tt', 
    #     'snack', 'Lunch', 'tea', 'desert', 'ted'
    #  }
    df['event'] = df['event'].str.lower()
    df['event'] = df['event'].replace({
        'diner': 'dinner',
        'lunb': 'lunch',
        })

    # we'll only take these 4 events, since this is most of the data
    df = df.drop(df[~df["event"].isin(['breakfast', 'lunch', 'dinner', 'workout'])].index)

    # drop rows that are missing data we need
    df = df.dropna(subset=['netCarbs', 'recomendedInsulinAmount'])

    ### We need to fix the time. A lot of the time are all AM because my Dad got lazy.
    ### We can assume AM/PM based off the event.
    # "Wednesday, August 13, 2025",breakfast,8:16am,33.97,5.90,3.98,0.00
    # "Wednesday, August 13, 2025",lunch,11:27am,43.76,6.20,7.13,0.00
    # "Wednesday, August 13, 2025",dinner,6:15am,73.40,13.00,11.27,0.00
    # "Wednesday, August 13, 2025",workout,9:39am,18.74,9.40,2.46,0.00
    def fix_time(row: pd.Series):

        event    : str = row['event']
        eventTime: str = row['eventTime'].lower()
        eventDate: str = row['date']

        hour, _, minute = eventTime.partition(":")

        if event == 'breakfast':
            eventTime = eventTime.replace('pm', 'am')

        elif event == 'dinner' or event == 'workout':

            # 12 would be a very late dinner
            if int(hour) != 12:
                eventTime = eventTime.replace('am', 'pm')

        elif event == 'lunch':

            # late lunch
            if int(hour) == 12 or int(hour) < 6:
                eventTime = eventTime.replace('am', 'pm')

            # otherwise assume early lunch
            else:
                eventTime = eventTime.replace('pm', 'am')

        return datetime.strptime(
                eventDate + " " + eventTime, 
                "%A, %B %d, %Y %I:%M%p"
            )

    df['dateTime'] = df[['event', 'eventTime', 'date']].apply(fix_time, axis=1)

    def fix_actual_insulin(row: pd.Series):
        
        rec: float = row['recomendedInsulinAmount']
        act: float = row['actualInsulinAmount']

        # My Dad's insulin pen only can go up to .5 accuracy, so we fill in missing taken amounts based off of this.
        if act == 0:

            act = round(rec * 2) / 2

        return act

    df['insulinTaken'] = df[['recomendedInsulinAmount', 'actualInsulinAmount']].apply(fix_actual_insulin, axis=1)
    df['insulinRec'] = df['recomendedInsulinAmount']

    # day,eventTime,event,bloodGlucose,netCarbs,insulinToCarbRatio,actualInsulinAmount,recomendedInsulinAmount,insulinSensitivityFactor,bloodGlucoseTarget
    df = df.drop(columns=['date', 'eventTime', 'actualInsulinAmount'])
    df = df[[ "dateTime", "event", "netCarbs", "bloodGlucose", "insulinToCarbRatio", "insulinSensitivityFactor", "bloodGlucoseTarget", "insulinRec" ,"insulinTaken" ]]
    
    return df



def main():

    # day, event, eventTime, netCarbs, bloodGlucose, recomendedInsulinAmount, actualInsulinAmount
    df: pd.DataFrame = pd.read_csv('./data/home_insulin.csv')

    print(df)

    df = clean_data(df)
    df['targetBG'] = 6.0
    df.to_csv("./data/home_insulin_clean.csv", index=False)

    time_series_df = create_time_series(df)
    time_series_df.to_csv("./data/home_insulin_clean_timeseries.csv", index=False)

    target_gc_df = create_target_gc_df(df)
    target_gc_df.to_csv("./data/home_insulin_clean_target_gc.csv", index=False)



if __name__ == "__main__":
    main()

