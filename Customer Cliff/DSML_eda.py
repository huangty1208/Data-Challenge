
dd.dtypes

dd.info()



def clean_col_names(df):
    """
    Simple function to convert the column 
    names of a dataframe to snake_case and lower case.
    """
    # Get all the col names as lower and snake_case in a list
    new_col_names = [
        column.strip().replace(' ', '_').lower() for column in df.columns
    ]
    # Rename the column names
    df.columns = new_col_names

    return df

  
# if dtypes wrong
pd.to_numeric(dd[cols], errors='coerce')
dd['col1'] = df['col1'].astype('category')

# or 
# Setting as categorical with order

# Create the list of ordered categories
order = ['small', 'medium small', 'medium', 'medium large', 'large']
dd['col1'] = pd.Categorical(dd['col1'], categories=order, ordered=True)


# Set as datetime
dd['col_date'] = pd.to_datetime(dd['col_date'])
# Check the conversion with an assert statement
assert dd['col_date'].dtype == 'datetime64[ns]'

# duplicates
duplicates = dd.duplicated()

dd.drop_duplicates(inplace=True)




  
