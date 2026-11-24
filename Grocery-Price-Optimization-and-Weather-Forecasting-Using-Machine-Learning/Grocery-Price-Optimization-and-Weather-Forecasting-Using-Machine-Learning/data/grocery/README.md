# Grocery dataset

Place the authorized grocery sales dataset here as `grocery_sales.csv`.

Expected columns:
`Item_ID`, `Category`, `Sub_Category`, `Price`, `Quantity_Sold`,
`Day_of_Week`, `Store_ID`, and either `Sales_Category` or `Total_Sales`.

The project report says the target labels are Low / Medium / High based on
`Total_Sales`, but it does not specify the exact cut-off values. If your file
does not already contain `Sales_Category`, the training script uses quantile
groups as a documented fallback.
