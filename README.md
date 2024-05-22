# Prediction-Analysis on META, IBM, GE, PG AND APPLE ON REV, EARNINGS AND DEVIDENDS

# DATA CREDIT TO  SEC Edgar

This project aims to analyze companies revenue data using various curve fitting techniques, including linear regression, logarithmic, exponential, and power curve fitting. The objective is to identify the best-fitting curve to predict companies revenue, earnings and devidends based on quartile data.

## Project Overview

This repository contains Python scripts and data files for conducting curve fitting analysis on IBM revenue data. The analysis involves the following steps:

1. **Data Preparation**: Reading the companies revenue, earnings and devidends data from an Excel file and preprocessing it for analysis.

2. **Curve Fitting Analysis**: Implementing linear regression and different curve fitting techniques to fit curves to the data.

3. **Visualization**: Visualizing the results of curve fitting and saving the plots for further analysis.

## Project Structure
├── data
│ └── home_four_dataOne.xlsx
├── images
│ └── power_curve_fit.png
├── companies_revenue_analysis.py
└── README.md



## Setup

To set up the project environment, follow these steps:

1. Clone the repository:

git clone https://github.com/yourusername/comapnies-revenue-analysis.git
cd comapnies-revenue-analysis

## Data

Place your `home_four_dataOne.xlsx` file in the `data` directory.

## Running the Code

To execute the analysis, run the `ibm_revenue_analysis.py` script:


The script will perform the curve fitting analysis and generate plots for visualization.

## Output

The script will output the R^2 score for each curve fitting technique, indicating the goodness of fit of the models.

## Plots

The plot of the power curve fitting will be saved in the `images` directory.

## Example Output

for each individual code desired company 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
