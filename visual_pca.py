# !pip install plotly 
import plotly.express as px
import numpy as np 


exp_var_cumul = np.array(my_model.explained_variance_ratio_.cumsum())

px.area(
    x=range(1, exp_var_cumul.shape[0] + 1),
    y=exp_var_cumul,
    labels={"x": "# Components", "y": "Explained Variance"}
)