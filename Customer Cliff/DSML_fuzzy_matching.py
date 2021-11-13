import recordlinkage as rl

# Create an indexing object
indexer = rl.Index()

# Set the mode of generation to full
indexer.full()

pairs = indexer.index(rand_a, rand_b)

# results will be a andas.MultiIndex, with the product of the lengths of datasets
# use blocking on a similar column among two dataset to get a reduced product


# Set the mode to blocking with `state`
indexer.block('state')
# Generate pairs
pairs = indexer.index(rand_a, rand_b)


### full example

## set a timer
start = time.time()
# Create an indexing object
indexer = rl.Index()
# Block on state
indexer.block('state')
# Generate candidate pairs
pairs = indexer.index(census_a, census_b)

# Create a comparing object
compare = rl.Compare()

# Query the exact matches of state
compare.exact('state', 'state', label='state')
# Query the exact matches of date of birth
compare.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
# Query the exact matches of date of birth
compare.exact('soc_sec_id', 'soc_sec_id', label='soc_sec_id')
# Query the exact matches of date of birth
compare.exact('postcode', 'postcode', label='postcode')
# Query the fuzzy matches for given name
compare.string('given_name', 'given_name', threshold=0.75, 
                method='levenshtein', label='given_name')
# Query the fuzzy matches for surname
compare.string('surname', 'surname', threshold=0.75, 
                method='levenshtein', label='surname')
# Query the fuzzy matches for address
compare.string('address_1', 'address_1', threshold=0.75, 
                method='levenshtein', label='address')

# Compute the matches, this will take a while
matches = compare.compute(pairs, census_a, census_b)
# Query matches with score over 4
full_matches = matches[matches.sum(axis='columns') >= 4]

# Get the indexes from either of index levels
duplicates = full_matches.index.get_level_values('rec_id_2')
# Exclude the indexes of duplicates from census_b
unique_b = census_b[~census_b.index.isin(duplicates)]

# Append deduplicated census_b to census_a
full_census = census_a.append(unique_b)

# end timer
end = time.time()
