import pandas as pd

def output_val(input_df,name):
    print(name)
    input_df.to_excel(name + '.xlsx')

def test_val(input_data, output_data):
    # Create DataFrames from input and output data
    df_input = pd.DataFrame(input_data)
    df_output = pd.DataFrame(output_data)

    # Print the original input DataFrame
    print("Input DataFrame:")
    print(df_input)

    # Print the output DataFrame
    print("\nOutput DataFrame:")
    print(df_output)

    # Compare the input and output DataFrames
    print("\nComparison of Input and Output DataFrames:")
    for col in df_input.columns:
        if col in df_output.columns:
            # Ensure column data types match for comparison
            if df_input[col].dtype != df_output[col].dtype:
                df_output[col] = df_output[col].astype(df_input[col].dtype)

            is_equal = df_input[col].equals(df_output[col])
            print(f"Column '{col}': {'Identical' if is_equal else 'Different'}")
        else:
            print(f"Column '{col}' is missing in output DataFrame.")

    # Check if there are extra columns in the output DataFrame not present in input
    for col in df_output.columns:
        if col not in df_input.columns:
            print(f"Extra column '{col}' found in output DataFrame.")
