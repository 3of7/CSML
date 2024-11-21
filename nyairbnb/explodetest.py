import pandas as pd
import re
listings = pd.read_csv('listings.csv', usecols=['id', 'amenities'])
listings.amenities = listings.amenities.replace('\[\]', '')
print(listings)

def split_quoted_list(lst):
    # Join the list into a single string and remove the surrounding brackets
    lst = re.sub(r'\[|\]', '', lst).split(', ')
    amenities = [re.sub(r'"', '', item) for item in lst ]
    return amenities

listings['amenities'] = listings['amenities'].apply(split_quoted_list)
listings.set_index('id', inplace=True)
exploaded = listings.explode('amenities')
dummies = pd.get_dummies(exploaded['amenities'])
result = dummies.groupby(exploaded.index).sum()
result = result.reset_index()
df2 = pd.read_csv('df2.csv')
result = result[result['id'].isin(df2['id'])]

final_df = df2.join(result.set_index('id'), on='id', how='left')
print(final_df.shape)
final_df = final_df.drop(final_df.filter(regex=".*inch HDTV.*").columns, axis=1)
final_df = final_df.drop(final_df.filter(regex=".*inch TV.*").columns, axis=1)
final_df = final_df.drop(final_df.filter(regex=".*Wifi.*").columns, axis=1)

non_differntiating = ['Carbon monoxide alarm','Dishes and silverware', 'Cooking basics', 'Essentials',
                      'Fire extinguisher', 'Hair dryer', 'Hangers', 'Hot water', 'Iron', 'Kitchen', 'Microwave',
                      'Refrigerator', 'Refrigerator refrigerator', 'Blomberg refrigerator', 'Self check-in', 'Smoke alarm'
                      ]
final_df = final_df.drop(non_differntiating, axis=1)
print(final_df.shape)
print(final_df.corr()['actual_price'].sort_values(ascending=False))
final_df=final_df.iloc[:, 1:]

final_df.drop(final_df.columns[41:42], axis=1,inplace=True)
final_df.to_csv('final_df.csv')
(final_df.corr()['actual_price'].sort_values(ascending=False)).to_csv('corr.csv')

