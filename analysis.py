# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
from collections import Counter
from google.colab import files

# Prompt to upload CSV files in Google Colab
print("Please upload the users.csv file.")
uploaded_users = files.upload()
print("Please upload the repositories.csv file.")
uploaded_repositories = files.upload()

# Load the uploaded CSV files
users_file_name = list(uploaded_users.keys())[0]
repositories_file_name = list(uploaded_repositories.keys())[0]
users_df = pd.read_csv(users_file_name)
repositories_df = pd.read_csv(repositories_file_name)

# Step 1: Clean and process user data
def clean_company_name(company):
    return company.strip().lstrip('@').upper() if pd.notna(company) else ""

users_df['company'] = users_df['company'].apply(clean_company_name)

# Step 2: Filter Paris users with over 200 followers
paris_users = users_df[(users_df['location'].str.contains('Paris', case=False, na=False)) & (users_df['followers'] > 200)]

# Question 1: Top 5 users in Paris with the highest followers
top_5_paris_users = paris_users.nlargest(5, 'followers')
print("Top 5 users in Paris by followers:", ', '.join(top_5_paris_users['login']))

# Question 2: 5 earliest registered GitHub users in Paris
users_df['created_at'] = pd.to_datetime(users_df['created_at'], errors='coerce')
earliest_users = users_df[users_df['location'].str.contains('Paris', case=False, na=False)].nsmallest(5, 'created_at')
print("5 earliest registered GitHub users in Paris:", ', '.join(earliest_users['login']))

# Question 3: Top 3 licenses among these users
popular_licenses = repositories_df['license_name'].value_counts().head(3)
print("Top 3 popular licenses:", ', '.join(popular_licenses.index))

# Question 4: Most common company for these developers
top_company = users_df['company'].value_counts().idxmax()
print("Most common company among these developers:", top_company)

# Question 5: Most popular programming language
popular_language = repositories_df['language'].value_counts().idxmax()
print("Most popular programming language:", popular_language)

# Question 6: Second most popular language among users who joined after 2020
recent_users = users_df[users_df['created_at'] > '2020-01-01']
recent_logins = recent_users['login']
second_popular_language = repositories_df[repositories_df['login'].isin(recent_logins)]['language'].value_counts().nlargest(2).idxmin()
print("Second most popular language for users after 2020:", second_popular_language)

# Question 7: Language with highest average stars per repository
avg_stars_per_language = repositories_df.groupby('language')['stargazers_count'].mean().idxmax()
print("Language with highest average stars per repo:", avg_stars_per_language)

# Question 8: Top 5 users by leader_strength
users_df['leader_strength'] = users_df['followers'] / (1 + users_df['following'])
top_5_leader_strength = users_df.nlargest(5, 'leader_strength')['login']
print("Top 5 users by leader_strength:", ', '.join(top_5_leader_strength))

# Question 9: Correlation between followers and public repositories
correlation = users_df['followers'].corr(users_df['public_repos'])
print("Correlation between followers and public repos:", correlation)

# Question 10: Regression estimate for additional followers per public repo
X = sm.add_constant(users_df['public_repos'])
Y = users_df['followers']
model = sm.OLS(Y, X).fit()
print("Regression summary for followers vs public repos:\n", model.summary())

# Question 11: Correlation between projects and wiki enabled
correlation_projects_wiki = repositories_df['has_projects'].astype(int).corr(repositories_df['has_wiki'].astype(int))
print("Correlation between projects and wiki enabled:", correlation_projects_wiki)

# Question 12: Comparison of following between hireable and non-hireable users
hireable_following_avg = users_df[users_df['hireable'] == True]['following'].mean()
non_hireable_following_avg = users_df[users_df['hireable'] == False]['following'].mean()
print("Avg following for hireable users:", hireable_following_avg)
print("Avg following for non-hireable users:", non_hireable_following_avg)

# Question 13: Regression on bio length and followers
users_df['bio_word_count'] = users_df['bio'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
X = sm.add_constant(users_df[users_df['bio_word_count'] > 0]['bio_word_count'])
Y = users_df[users_df['bio_word_count'] > 0]['followers']
bio_model = sm.OLS(Y, X).fit()
print("Bio length regression summary:\n", bio_model.summary())

# Question 14: Top 5 users who created most repositories on weekends
repositories_df['created_at'] = pd.to_datetime(repositories_df['created_at'], errors='coerce')
repositories_df['day_of_week'] = repositories_df['created_at'].dt.dayofweek
weekend_repos = repositories_df[repositories_df['day_of_week'] >= 5]
top_weekend_creators = weekend_repos['login'].value_counts().head(5)
print("Top 5 users creating most repos on weekends:", ', '.join(top_weekend_creators.index))

# Question 15: Hireable users sharing email more often
total_hireable = users_df['hireable'].sum()
total_non_hireable = len(users_df) - total_hireable
email_hireable = users_df[users_df['hireable'] == True]['email'].notna().sum() / total_hireable if total_hireable > 0 else 0
email_non_hireable = users_df[users_df['hireable'] == False]['email'].notna().sum() / total_non_hireable if total_non_hireable > 0 else 0
print("Email sharing by hireable users:", email_hireable)
print("Email sharing by non-hireable users:", email_non_hireable)

# Question 16: Most common surname(s) and count
users_df['surname'] = users_df['name'].apply(lambda x: x.split()[-1] if pd.notna(x) and len(x.split()) > 1 else None)
surname_counts = Counter(users_df['surname'].dropna())
max_count = max(surname_counts.values())
most_common_surnames = [surname for surname, count in surname_counts.items() if count == max_count]
print("Most common surname(s):", ', '.join(most_common_surnames))
print("Count for the most common surname:", max_count)
