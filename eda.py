import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 1. CREATE RAW DATAFRAME
np.random.seed(42)
cities = [' New York', 'Los Angeles', 'Chicago', 'Housston', 'San Francisco'] 
house_types = ['Apartment', 'Condo', 'Villa', 'Townhouse']
dates = [datetime(2010, 1, 1) + timedelta(days=random.randint(0, 5000)) for _ in range(1000)]


df = pd.DataFrame({
    'City': np.random.choice(cities, 1000),
    'House_Type': np.random.choice(house_types, 1000),
    'Built_Year': np.random.randint(1950, 2023, 1000),
    'Price': np.random.randint(50000, 1000000, 1000),
    'Area_sqft': np.random.randint(500, 5000, 1000),
    'Bedrooms': np.random.randint(1, 6, 1000),
    'Bathrooms': np.random.randint(1, 5, 1000),
    'Garage': np.random.choice(['Yes', 'No'], 1000),
    'Listing_Date': np.random.choice(dates, 1000),
    'Agent_Name': np.random.choice([' John Doe', 'Jane Smith', 'Mike Brown'], 1000)
})

print(df)

# Add missing values randomly
for col in df.columns:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

print(df)

# 2. HANDLE MISSING VALUES
print(df['Price'].isnull().sum())
df['Price'].fillna(df['Price'].median(), inplace=True)
print(df['Price'])
print(df['Price'].isnull().sum())

print(df['Area_sqft'].isnull().sum())
df['Area_sqft'].fillna(df['Area_sqft'].mean(), inplace=True)
print(df['Area_sqft'])
print(df['Area_sqft'].isnull().sum())

df['City'].isnull().sum()
df['City'].fillna('Unknown', inplace=True)
df.dropna(subset=['Listing_Date'], inplace=True)  # Drop if no listing date
df['City'].isnull().sum()


# 3. REMOVE DUPLICATES
print(df.drop_duplicates(inplace=True))
# 4. FIX INCONSISTENT DATA TYPES
print(df['Built_Year'])
print(df['Price'])
print(df['Area_sqft'])

print(df.info())

#df['Built_Year'] = df['Built_Year'].astype('int32')
#df['Price'] = df['Price'].astype(float)
#df['Area_sqft'] = df['Area_sqft'].astype(float)

print(df)

# 5. STRIP WHITESPACE
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

print(df)   
# 6. CORRECT TYPOS IN CATEGORICAL DATA
df['City'] = df['City'].replace({'Housston': 'Houston'})
print(df['City'])

# 7. REMOVE OUTLIERS (IQR method for Price)
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Price'] >= Q1 - 1.5*IQR) & (df['Price'] <= Q3 + 1.5*IQR)]
print(f'Q1: {Q1}, Q3: {Q3}, IQR: {IQR}')
print(df)

# 8. NORMALIZE/ SCALE NUMERIC VALUES
scaler = MinMaxScaler()
df[['Price', 'Area_sqft']] = scaler.fit_transform(df[['Price', 'Area_sqft']])
print(df[['Price', 'Area_sqft']])

# 9. ENCODE CATEGORICAL VARIABLES
le = LabelEncoder()
df['City_Code'] = le.fit_transform(df['City'])
df['House_Type_Code'] = le.fit_transform(df['House_Type'])
print(df[['City', 'City_Code', 'House_Type', 'House_Type_Code']])


# 10. FEATURE EXTRACTION (House Age)
current_year = datetime.now().year
df['House_Age'] = current_year - df['Built_Year']
print(df[['Built_Year', 'House_Age']])


# 11. HANDLE DATE/TIME FIELDS
print(df['Listing_Date'])
df['Listing_Year'] = pd.to_datetime(df['Listing_Date']).dt.year
print(df['Listing_Year'])

# 12. FILL OR DROP ZERO VALUES
print(df[df['Area_sqft'] <= 0])

# 13. REORDER COLUMNS
df = df[['City', 'City_Code', 'House_Type', 'House_Type_Code', 'Built_Year', 'House_Age',
         'Price', 'Area_sqft', 'Bedrooms', 'Bathrooms', 'Garage', 'Listing_Date', 'Listing_Year', 'Agent_Name']]
print(df)
# 14. RENAME COLUMNS FOR CLARITY
df.rename(columns={'Area_sqft': 'Area_in_SqFt', 'Built_Year': 'Year_Built'}, inplace=True)
print(df.info())
# 15. DROP UNNECESSARY COLUMNS (example: Agent_Name)
df.drop(columns=['Agent_Name'], inplace=True)
print(df.info())

# 16. DETECT DATA IMBALANCE
imbalance = df['City'].value_counts(normalize=True)
print("\nData imbalance:\n", imbalance)

# 17. REMOVE NON-PRINTABLE CHARACTERS
df = df.applymap(lambda x: ''.join(c for c in x if c.isprintable()) if isinstance(x, str) else x)
print(df)

# 18. HANDLE NEGATIVE VALUES (where not logical)
num_cols = ['Price', 'Area_in_SqFt', 'Bedrooms', 'Bathrooms', 'House_Age']
for col in num_cols:
    df = df[df[col] >= 0]

# 19. CHECK FOR UNIQUE VALUES
print("\nUnique value counts:\n", df.nunique())

# 20. BACKUP CLEANED DATA
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"cleaned_house_price_{timestamp}.csv", index=False)
print("\n✅ Cleaned data backed up to CSV")

# Final dataset preview
#print("\nCleaned DataFrame:\n", df.head())
#________________________________________
#What I Does
#•	Generates synthetic house price data with realistic columns.
#•	Intentionally injects missing values, typos, whitespace.
#•	Applies all 20 cleaning steps in logical order.
#•	Produces a cleaned CSV backup with a timestamp.
#________________________________________
#If you want, I can add random anomalies like negative prices, extra spaces, and special characters so the cleaning process is more visibly effective.
#That way, you’ll see the before/after transformation clearly. Would you like me to make that enhanced version?



