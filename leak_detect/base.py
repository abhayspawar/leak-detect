import pandas as pd
import numpy as np

def get_nan_counts(data, cols, null_col_suffix=''):
    """
    Returns dataframe containing number of nulls in 'cols' in 'data'

        Parameters:
            data (dataframe): input data with 'cols' in it.
            cols (list): columns used for computing number of nulls in them.
            null_col_suffix (string): suffix used for null_counts column in returned dataframe.

        Returns: 
            nulls_df (dataframe): contains null counts for columns 'cols' in 'data'.

    """
    nulls_df = pd.DataFrame(pd.isnull(data[cols]).sum())
    nulls_df.columns = ['null_counts'+null_col_suffix]
    nulls_df['feature'] = nulls_df.index
    nulls_df.reset_index(inplace=True, drop=True)
    return nulls_df

def detect_vertical_leakage_from_to(data_creation_func, input_data, input_feature_cols, output_feature_cols, 
                                      use_nan=True, check_row_number=-1, direction='upward'):
    """
    Checks if vertical leakage is happening in 'output_feature_cols' columns while creating them from 'input_feature_cols'.
    Prints out 'output_feature_cols' which have vertically leaked data. Can check for upward (default) or downward leakage.
    Gets called by wrapper function 'detect_vertical_leakage'.

            Parameters:
            data_creation_func (function): data creation function which takes in 'input_data' and 
                returns a dataframe containing 'output_feature_cols' created from 'input_feature_cols'.
            input_data (dataframe): containing 'input_feature_cols' and used as input to 'data_creation_func'.
            input_feature_cols (list): columns in input data which are used to compute 'output_feature_cols' in 
                'data_creation_func'.
            output_feature_cols (list): columns created through 'data_creation_func' for which we want to check if data has 
                leaked vertically in given direction.
            use_nan (boolean): If NANs should be used. Complex numbers are used if set to False. Default is NANs.
            check_row_number (int): row number above or below which all rows are set to NANs/complex numbers depending 
                on check direction. By default, set to be at midway: int(len(data)/2).
            direction (str): Direction in which leakage is to be detected. 'upward' implies 'output_feature_cols' will 
                be checked for leakage from later rows into previous rows. If rows are sorted by date, you would want 
                the data from a row to not flow into previous rows. So, the check should be for 'upward' direction.
                'downward' implies in the other direction. Default is 'upward'.

            Returns:
                has_leakage (boolean): True if leakage is happening vertically in given direction. Else false.

    """

    input_data_feats = data_creation_func(input_data)

        
    for col in output_feature_cols:
        if col not in input_data_feats.columns:
            raise Exception("Column {} in 'output_feature_cols' is not present in the data created ".format(col) +
                            "using 'data_creation_func'")
        
    if direction=='upward':
        nulls_df_orig = get_nan_counts(input_data_feats[:check_row_number], output_feature_cols)
        start = check_row_number
        end = len(input_data)
        print_direction = 'previous'
    else:
        nulls_df_orig = get_nan_counts(input_data_feats[check_row_number:], output_feature_cols)
        start = 0
        end = check_row_number
        print_direction = 'next'

    
    # for df.loc last element is included when returning subset of rows
    if use_nan:
        input_data.loc[start:end-1, input_feature_cols] = np.nan
    else:
        for col in input_feature_cols:
            input_data.loc[start:end-1, col] = input_data.loc[start:end-1, col].apply(lambda x: np.complex(x + 1j))

    input_data_feats_null = data_creation_func(input_data)
    if not use_nan:
        # pretend like NANs are leaking whenever complex number leaks
        for col in output_feature_cols:
            input_data_feats_null[col] = input_data_feats_null[col].apply(lambda x: np.nan if np.iscomplex(x) else x)
    
    if len(input_data_feats)!=len(input_data_feats_null):
        print("WARNING! Number of rows in features data created with and without NANs/complex number are different.",
              "Please do not drop any null rows in 'data_creation_func'. These results aren't reliable!")

    #upward: 0 to check_row_number else (check_row_number to end)
    (start, end) = (0, check_row_number) if start==check_row_number else (check_row_number, len(input_data_feats_null))
        
    nulls_df_detect = get_nan_counts(input_data_feats_null[start:end], output_feature_cols, 
                                     null_col_suffix='_detect')
    nulls_df_orig = nulls_df_orig.merge(nulls_df_detect, on='feature', how='left')
    leaky_features = nulls_df_orig[nulls_df_orig['null_counts_detect']!=nulls_df_orig['null_counts']]

    has_leakage = False
    if len(leaky_features)>0:
        has_leakage = True
        print('Oops vertical leakage detected!!')
        print('List of columns and number of {} rows into which data is'.format(print_direction),
              'leaking from a row:')
        for i in range(len(leaky_features)):
            curr_record = leaky_features[i:i+1]
            feature = curr_record['feature'].values[0]
            data_leakage_count = curr_record['null_counts_detect'].values[0] - curr_record['null_counts'].values[0]
            print(feature, ':', int(data_leakage_count))

    if not has_leakage:
        print('No vertical leakage detected. Good to go! Yay!!')

    print('\n')
    return has_leakage

def detect_vertical_leakage(data_creation_func, input_data, input_feature_cols, output_feature_cols, only_nan=False,
                            check_row_number=-1, direction='upward'):

    """
    Checks if vertical leakage is happening in 'output_feature_cols' columns while creating them from 'input_feature_cols'
    by using NANs (and complex numbers if only_nan=False).
    Prints out 'output_feature_cols' which have vertically leaked data. Can check for upward (default) or downward leakage.
    Wrapper function for 'detect_vertical_leakage_from_to' to compute leakage using NANs and complex numbers.

        Parameters:
            data_creation_func (function): data creation function which takes in 'input_data' and 
                returns a dataframe containing 'output_feature_cols' created from 'input_feature_cols'.
            input_data (dataframe): containing 'input_feature_cols' and used as input to 'data_creation_func'.
            input_feature_cols (list): columns in input data which are used to compute 'output_feature_cols'.
            output_feature_cols (list): columns created in 'data_creation_func' for which we want to check leakage
            only_nan (boolean): If only NANs should be used. By default, complex numbers are also used separately. 
            check_row_number (int): row number above or below which all rows are set to NANs/complex numbers depending 
                on check direction. By default, set to be at midway: int(len(data)/2).
            direction (str): Direction in which leakage is to be detected. 'upward' implies 'output_feature_cols' will 
                be checked for leakage from later rows into previous rows. If rows are sorted by date, you would want 
                the data from a row to not flow into previous rows. So, the check should be for 'upward' direction.
                'downward' implies in the other direction. Default is 'upward'.

        Returns: 
            has_leakage (boolean): True if leakage is happening vertically in given direction. Else false.

    """

    if direction not in ['upward', 'downward']:
        raise Exception("Please provide 'upward' or 'downward' as a value for 'direction' input")
    for col in input_feature_cols:
        if col not in input_data.columns:
            raise Exception("Column {} in 'input_feature_cols' is not present in the input 'data'".format(col))
    
    if check_row_number == -1:
        check_row_number = int(len(input_data)/2)
    if check_row_number>=len(input_data)-1:
        raise Exception("Please enter a value for 'check_row_number' lower than number of rows "+
                        "in the input data")

    has_leakage = 0
    print('Checking for vertical leakage in {} direction...'.format(direction))
    row_direction = 'after' if direction=='upward' else 'before'
    print("By replacing 'input_feature_cols' {} row number {} with NANs:".format(row_direction, check_row_number))
        
    data_in = input_data.copy()
    has_leakage += detect_vertical_leakage_from_to(data_creation_func, data_in, input_feature_cols, 
                                                   output_feature_cols, direction=direction, 
                                                   check_row_number=check_row_number)

    if not only_nan:
        data_in = input_data.copy()
        print("By adding imaginary component to 'input_feature_cols' {} row number {}:".format(row_direction, 
                                                                                     check_row_number))

        has_leakage += detect_vertical_leakage_from_to(data_creation_func, data_in, input_feature_cols, 
                                                       output_feature_cols, use_nan=False, direction=direction, 
                                                       check_row_number=check_row_number)
        
    return has_leakage>0

def detect_horizontal_leakage_from_to(data_creation_func, input_data, leakage_from_cols, leakage_to_cols, 
                                      use_nan=True):
    """
    Checks if leakage is happening from 'leakage_from_cols' columns to 'leakage_to_cols' columns when they 
    are computed in 'data_creation_func'.
    Prints out 'leakage_to_cols' which have leaked data from 'leakage_from_cols' and number of 
    rows with leakage. 
    Gets called by wrapper function 'detect_horizontal_leakage'.
    
        Parameters:
            data_creation_func (function): data creation function which takes in 'input_data' and 
                returns a dataframe containing 'leakage_to_cols' columns.
            input_data (dataframe): containing 'leakage_to_cols' and used as input to 'data_creation_func'.
            leakage_from_cols (list): columns in input data for which we want to check if their data is 
                leaking into 'leakage_to_cols'.
            leakage_to_cols (list): columns created in 'data_creation_func' for which we want to check if 
                data is leaking from 'leakage_from_cols'.
            use_nan (boolean): If NANs should be used. Complex numbers are used if False. Default is True.

        Returns:
            has_leakage (boolean): True if leakage is happening from 'leakage_from_cols' to 'leakage_to_cols'.

    """
    
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

    if len(input_data_feats)!=len(input_data_feats_null):
        print("WARNING! Number of rows in features data created with and without NANs/complex number are different.",
              "Please do not drop any null rows in 'data_creation_func'. These results aren't reliable!")

    nulls_df_detect = get_nan_counts(input_data_feats_null, cols=leakage_to_cols, null_col_suffix='_detect')

    nulls_df_orig = nulls_df_orig.merge(nulls_df_detect, on='feature', how='left')
    leaky_features = nulls_df_orig[nulls_df_orig['null_counts_detect']!=nulls_df_orig['null_counts']]
    
    has_leakage = False
    if len(leaky_features)>0:
        has_leakage = True
        print('Oops horizontal leakage detected!! \nList of columns and their respective number of rows with leaky data:')
        for i in range(len(leaky_features)):
            curr_record = leaky_features[i:i+1]
            feature = curr_record['feature'].values[0]
            data_leakage_count = curr_record['null_counts_detect'].values[0] - curr_record['null_counts'].values[0]
            print(feature, ':', int(data_leakage_count))
    
    if not has_leakage:
        print('No horizontal leakage detected. Good to go! Yay!!')
        
    print('\n')
        
    return has_leakage

def detect_horizontal_leakage(data_creation_func, input_data, target_cols, output_feature_cols, input_feature_cols=[], 
                              only_nan=False):
    """
    Checks if leakage is happening from 'target_cols' columns to 'output_feature_cols' columns when they are computed
    in 'data_creation_func'.
    Prints out 'output_feature_cols' which have leaked data from 'target_cols' and number of rows with leakage.
    If 'input_feature_cols' are passed, then also checks leakage from these to 'target_cols'.

        Parameters:
            data_creation_func (function): data creation function which takes in 'input_data' and 
                returns a dataframe containing 'output_feature_cols' columns.
            input_data (dataframe): must contain pre-computed 'target_cols'. If 'input_feature_cols' are passed, they
                are required to be there in 'input_data'.
            target_cols (list): List containing dependent variables' column name.
            output_feature_cols (list): columns created in 'data_creation_func' for which we want to check if data 
                is leaking from 'target_cols'.
            input_feature_cols (list): columns from 'input_data' used in 'data_creation_func' to create features data. 
                We want to check if data is leaking from these columns into 'target_cols'. 
                By deafult, no columns are passed and this test is not done.
            only_nan (boolean): Only NANs are used for check if True. Complex numbers are also used if False.
                Default False.

        Returns:
            has_leakage (boolean): True if leakage is happening from 'target_cols' to 'output_feature_cols' or from
                'input_feature_cols' to 'target_cols'.
                
    """
    # Needs the target columns to already present in input data and shouldnt be recreated in data_creation_func
    has_leakage = 0

    print('Checking for leakage from target columns to feature columns...')
    print('By replacing target with NANs:')   
    data_in = input_data.copy()
    has_leakage += detect_horizontal_leakage_from_to(data_creation_func, data_in, target_cols, output_feature_cols)

    if not only_nan:
        data_in = input_data.copy()
        print('By adding imaginary component to target columns:')
        has_leakage += detect_horizontal_leakage_from_to(data_creation_func, data_in, target_cols, 
                                                         output_feature_cols, use_nan=False)
    
    if input_feature_cols:
        print('---------------------------------------------------------------------------')

        print('Checking for leakage from input feature columns to target columns...')
        print('By replacing input feature columns with NANs:')        
        data_in = input_data.copy()
        has_leakage += detect_horizontal_leakage_from_to(data_creation_func, data_in, input_feature_cols, 
                                                         target_cols)
        
        if not only_nan:
            print('By adding imaginary component to input feature columns:')
            data_in = input_data.copy()
            has_leakage += detect_horizontal_leakage_from_to(data_creation_func, data_in, input_feature_cols, 
                                                             target_cols, use_nan=False)
    
    return has_leakage>0
