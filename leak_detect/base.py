import pandas as pd
import numpy as np

def get_nan_counts(data, leakage_to_cols, null_col_suffix=''):
    nulls_df = pd.DataFrame(pd.isnull(data[leakage_to_cols]).sum())
    nulls_df.columns = ['null_counts'+null_col_suffix]
    nulls_df['feature'] = nulls_df.index
    nulls_df.reset_index(inplace=True, drop=True)
    return nulls_df

def detect_horizontal_leakage_from_to(data_creation_func, input_data, leakage_from_cols, leakage_to_cols, 
                                      use_nan=True):
    # checks for leakage from 'from' cols to 'to' cols. Gets called by main horizontal leakage function.
    # By default uses NANs for leakage detection.
    
    input_data_feats = data_creation_func(input_data)
    nulls_df_orig = get_nan_counts(input_data_feats, leakage_to_cols)

    if use_nan:
        input_data.loc[:, leakage_from_cols] = np.nan
    else:
        for col in leakage_from_cols:
            input_data.loc[:, col] = input_data[col].apply(lambda x: np.complex(x + 1j))

    input_data_feats_null = data_creation_func(input_data)
    if not use_nan:
        # pretend like NANs are leaking whenever complex number leaks
        for col in leakage_to_cols:
            input_data_feats_null[col] = input_data_feats_null[col].apply(lambda x: np.nan if np.iscomplex(x) else x)

    nulls_df_detect = get_nan_counts(input_data_feats_null, leakage_to_cols=leakage_to_cols, null_col_suffix='_detect')

    nulls_df_orig = nulls_df_orig.merge(nulls_df_detect, on='feature', how='left')
    leaky_features = nulls_df_orig[nulls_df_orig['null_counts_detect']!=nulls_df_orig['null_counts']]
    
    has_leakage = False
    if len(leaky_features)>0:
        has_leakage = True
        print('Oops horizontal leakage detected!! \nList of columns and their respective rows with leaky data:')
        for i in range(len(leaky_features)):
            curr_record = leaky_features[i:i+1]
            feature = curr_record['feature'].values[0]
            data_leakage_count = curr_record['null_counts_detect'].values[0] - curr_record['null_counts'].values[0]
            print(feature, ':', int(data_leakage_count))
    
    if not has_leakage:
        print('No horizontal leakage detected. Good to go! Yay!!')
        
    print('\n')
        
    return has_leakage

def detect_horizontal_leakage(data_creation_func, data, target_cols, output_feature_cols, input_feature_cols=[], 
                              only_nan=False):
    
    # Main function for horizontal leakage. Checks for two leakage from target cols to feature cols and vice versa 
    # if input_feature_cols are provided. By default runs checks using both nans and comples numbers.
    # Needs the target columns to already present in input data and shouldnt be recreated in data_creation_func
    # input function.
    # Returns True is leakage is present.
    # Lists out columns with leakage and number rows with leakage for quicker debugging
    has_leakage = 0

    print('Checking for leakage from target columns to feature columns...')
    print('By replacing target with NANs:')   
    data_in = data.copy()
    has_leakage += detect_horizontal_leakage_from_to(data_creation_func, data_in, target_cols, output_feature_cols)

    if not only_nan :
        data_in = data.copy()
        print('By adding imaginary component to target columns:')
        has_leakage += detect_horizontal_leakage_from_to(data_creation_func, data_in, target_cols, 
                                                         output_feature_cols, use_nan=False)
    
    if input_feature_cols:
        print('---------------------------------------------------------------------------')

        print('Checking for leakage from input feature columns to target columns...')
        print('By replacing input feature columns with NANs:')        
        data_in = data.copy()
        has_leakage += detect_horizontal_leakage_from_to(data_creation_func, data_in, input_feature_cols, 
                                                         target_cols)
        
        if not only_nan:
            print('By adding imaginary component to input feature columns:')
            data_in = data.copy()
            has_leakage += detect_horizontal_leakage_from_to(data_creation_func, data_in, input_feature_cols, 
                                                             target_cols, use_nan=False)
    
    return has_leakage>0