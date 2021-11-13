
# Convert column to categorical
demographics['marriage_status'] = demographics['marriage_status'].astype(
    'category')
# Check the results
assert demographics['marriage_status'].dtype == 'category'


# Collapsing Data Into Categories using cut

ranges = [40000, 75000, 100000, 
          140000, 170000, np.inf] # infinity
labels = ['40k-75k', '75k-100k', '100k-140k', 
          '140k-170k', '170k+']
demographics['income_groups'] = pd.cut(demographics['income'], 
                                       bins=ranges, 
                                       labels=labels)


# Reducing the Number of Categories

mappings = {
    'Linux': 'desktopOS', 'iOS': 'mobileOS',
    'MacOS': 'desktopOS', 'Windows': 'desktopOS',
    'AndroidOS': 'mobileOS'
}

emographics['device'] = demographics['device'].replace(mappings)


# pivot_table returns a dataframe

# Using groupby
result = tips.groupby('sex')['total_bill'].sum()
type(result)
pandas.core.series.Series


# Using pivot_table
result_pivot = tips.pivot_table(values='total_bill', index='sex', aggfunc=np.sum)

# different dataframe output format
tips.groupby(['sex', 'day'])['total_bill']\
            .agg([np.mean, np.median, np.sum]).reset_index()

tips.pivot_table(values='total_bill', 
                 index=['sex', 'day'], 
                 aggfunc=[np.mean, np.median, np.sum])



