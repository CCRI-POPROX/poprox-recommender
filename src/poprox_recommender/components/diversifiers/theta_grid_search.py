import pandas as pd

test_request_data_path = '/home/sun00587/research/News_Locality_Polarization/Data/POPROX_data/1106/'

# Read in relevant log data
articles_df = pd.read_parquet(test_request_data_path+'articles_20241106-224642.parquet')
clicks_df = pd.read_parquet(test_request_data_path+'clicks_20241106-142158.parquet')
mentions_df = pd.read_parquet(test_request_data_path+'mentions_20241106-230121.parquet')
newsletters_df = pd.read_parquet(test_request_data_path+'newsletters_20241106-145025.parquet')

def construct_training_data():
    '''
        Construct training data for grid search recall and kl divergence calculation
        For each user, we have the top 10 articles recommended to them each day in the past 90 days
        
    '''


def metric_calculation():
    '''
        Recall@k = (# of recommended items @k that user clicked on) / (total # of relevant items [10])
    '''

