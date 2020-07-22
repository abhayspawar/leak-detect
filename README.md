# leak-detect
Leak-detect can be used to detect leakages in machine learning pipelines using complex numbers and NANs.

Detailed [blog post](https://towardsdatascience.com/detecting-data-leakage-in-ml-pipelines-using-nans-and-complex-numbers-66a066116b40) explaining the idea behind this package.
  
leak-detect contains two function to detect horizontal and vertical leakage in data creation pipelines.

`detect_horizontal_leakage_from_to(data_creation_func, input_data, leakage_from_cols, leakage_to_cols, 
                                      use_nan=True)
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
            has_leakage (boolean): True if leakage is happening from 'leakage_from_cols' to 'leakage_to_cols'.`
