age_df = pd.concat([age_df, app_train['TARGET']], axis=1)
# print(age_df.head())

age_df = age_df.sort_values(by=['AGE'])
value_counts = age_df['TARGET'].value_counts()
# print(value_counts)
total_non_events = value_counts[0]  # Total zeroes
total_events = value_counts[1]  # Total ones

age_bins = np.array_split(age_df, 10)  # Split into 10 equal bins
null_bin = pd.DataFrame()

# woe_df = pd.DataFrame(columns=['RANGE', 'BINS', 'NON_EVENTS', 'EVENTS', 'PERCENT_NON_EVENTS', 'PERCENT_EVENTS'])
woe_list = []  # Create list of dicts, then convert to DataFrame. Efficient because concat recreates data structure
bin_num = 0
for my_bin in age_bins:
    bin_num += 1

    # Not sure if these will catch NULL values ##PLEASE CHECK##---------------------------------------------------------
    null_bin = pd.concat([null_bin, my_bin.loc[my_bin['AGE'].isnull()]], ignore_index=True)
    # null_bin.append(my_bin.loc[my_bin['AGE'].isnull()])  # Append is supposedly slower than concat
    my_bin = my_bin.drop(my_bin[my_bin['AGE'].isnull()].index)  # Removes null values from bin

    wd = {}
    # print('Rows: ' + str(my_bin.shape[0]/app_train.shape[0]))  # % of all rows in this bin

    bin_min = my_bin['AGE'].min()  # Min bin
    bin_max = my_bin['AGE'].max()  # Max bin
    wd['RANGE'] = str(round(bin_min, 2)) + '-' + str(round(bin_max, 2))
    bin_value_counts = my_bin['TARGET'].value_counts()
    wd['NON_EVENTS'] = bin_value_counts[0]  # Non-Events in this bin
    wd['EVENTS'] = bin_value_counts[1]  # Events in this bin
    percent_non_events = wd['NON_EVENTS']/total_non_events
    percent_events = wd['EVENTS']/total_events
    wd['PERCENT_NON_EVENTS'] = str(round(percent_non_events*100, 2)) + '%'  # % of Non-Events
    wd['PERCENT_EVENTS'] = str(round(percent_events*100, 2)) + '%'  # % of Events
    woe = math.log(percent_non_events/percent_events)
    IV = (percent_non_events - percent_events) * woe
    wd['WOE'] = round(woe, 4)
    wd['IV'] = round(IV, 4)
    woe_list.append(wd)

woe_df = pd.DataFrame(woe_list, columns=['RANGE', 'NON_EVENTS', 'EVENTS', 'PERCENT_NON_EVENTS', 'PERCENT_EVENTS', 'WOE', 'IV'])
pd.set_option('display.expand_frame_repr', False)  # Allows DF to be displayed in full
# IV_total = woe_df['PSEUDO_IV'].sum()
woe_df.at['TOTAL', 'IV'] = woe_df['IV'].sum()
print(woe_df.head(11))
# print(null_bin.head())B
