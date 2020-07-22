# leak-detect
Leak-detect can be used to detect leakages in machine learning pipelines using complex numbers and NANs.

Detailed [blog post](https://towardsdatascience.com/detecting-data-leakage-in-ml-pipelines-using-nans-and-complex-numbers-66a066116b40) explaining the idea behind this package.
  
leak-detect contains two function to detect horizontal and vertical leakage in data creation pipelines. Description of input parameters and output for each function below.

### Vertical leakage detection

```
def detect_vertical_leakage(data_creation_func, input_data, input_feature_cols, output_feature_cols, only_nan=False,
                            check_row_number=-1, direction='upward'):


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

```


### Horizontal leakage detection

```
def detect_horizontal_leakage(data_creation_func, input_data, target_cols, output_feature_cols, input_feature_cols=[], 
                              only_nan=False)
                              
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
```
