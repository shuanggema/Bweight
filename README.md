# Bweight
Bayesian Modeling of Cancer Outcomes Using Genetic Variables Assisted by Pathological Imaging Data

# ---------------------------------------------------------------------
# Load required packages for data processing, plotting, and statistical analysis
# ---------------------------------------------------------------------
using Pkg, Plots, Plots.PlotMeasures, ColorSchemes, StatsPlots

# ---------------------------------------------------------------------
# Load and source external data and functions
# ---------------------------------------------------------------------
@load "simdata_surv.jld2"  # Load the pre-saved data file containing simulated data called "mydat"
include("/ftns-survival.jl")  # Include external functions for survival analysis (similar to source function in R)

# ---------------------------------------------------------------------
# Extract data from the loaded dataset
# ---------------------------------------------------------------------
y = mydat[:, "y"]  # Extract the response variable 'y' 
d = mydat[:, "d"]  # Extract the censoring indicator 'd' 
Z = Matrix(select(mydat, r"Z"))  # Select columns starting with 'Z' and convert them to a matrix (imaging outcomes)
X = Matrix(select(mydat, r"x"))  # Select columns starting with 'x' and convert them to a matrix (predictors)

# ---------------------------------------------------------------------
# Set parameters for MCMC sampling
# ---------------------------------------------------------------------
n_total, n_burn = 1000, 200  # Define the total number of MCMC iterations and the burn-in period
H = construct_H(X, Z) # Construct hyperparameters based on predictors (X) and imaging outcomes (Z)

# ---------------------------------------------------------------------
# Run the MCMC sampler
# ---------------------------------------------------------------------
s1 = run_sampler(y, X, Z, H; d = d, n_total = n_total, use_w = true); 
# Run the MCMC sampler using:
#   - Response variable (y)
#   - Predictors (X) and imaging outcomes (Z)
#   - Hyperparameters (H)
#   - Censoring indicator (d)
#   - Enable weights (use_w = true)

# ---------------------------------------------------------------------
# Analyze and visualize regression coefficients
# ---------------------------------------------------------------------
true_nz_id = [12, 16, 32, 33, 46, 55, 56, 70, 92, 99]  # Indices of true non-zero coefficients for visualization

# Compute and display the posterior mean of the selected coefficients, rounded to 2 decimal places
round.(mean(s1.b[(n_burn + 1) : n_total, true_nz_id], dims = 1), digits = 2)

# Create an empty plot for visualizing regression coefficients
myp = Plots.plot(size = (400, 300), ylab = "", fontsize = 15, legendfontsize = 4, grid = false) 
# Plot the trace plot of regression coefficients for the selected variables
Plots.plot!(s1.b[1: n_total, true_nz_id], legend = false) 

# Create an empty plot for visualizing regression coefficients
myp = Plots.plot(size = (400, 300), ylab = "", fontsize = 15, legendfontsize = 8, grid = false)
# Plot the coefficients for the first selected variable
legend_labels = ["γ₁,₁₂" "γ₂,₁₂" "γ₃,₁₂" "γ₄,₁₂" "γ₅,₁₂"]  # Labels for coefficients
Plots.plot!(s1.gs[1:n_total, true_nz_id[1], :], label = legend_labels)

# ---------------------------------------------------------------------
# Analyze and visualize the weights for the model
# ---------------------------------------------------------------------
Plots.scatter(
    mean(s1.wi[(n_burn + 1) : n_total, :], dims = 1)',  # Compute mean weights after burn-in period
    size = (400, 300),  label = false, grid = false)
