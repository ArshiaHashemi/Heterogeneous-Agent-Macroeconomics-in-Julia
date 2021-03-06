################################################################################
#
#   PURPOSE: Solve Aiyagari Model Using Endogenous Grid Points Method
#   AUTHOR: Arshia Hashemi
#   EMAIL: arshiahashemi@uchicago.edu
#   FIRST VERSION: 04/24/2022
#   THIS VERSION: 04/28/2022
#
################################################################################

## Preamble

# Working directory
cd(
    "C:\\Users\\arshi\\Dropbox\\ArshiaHashemiRA\\ECMA 33620\\assignments\\assignment3\\output\\",
);

# Packages
using DataFrames,
    CSV,
    Distributions,
    StatsBase,
    Optim,
    Random,
    Statistics,
    TexTables,
    IterableTables,
    LinearAlgebra,
    Colors,
    LaTeXStrings,
    Plots

## Parameters

# Preferences
Ξ³ = 1.5;
Ξ² = 0.95;

# efficiency units of labor tax
Ο = 0.15;

# Production
π΄ = 1.0;
Ξ± = 1 / 3;
Ξ΄ = 0.2;

# Effiency units of labor process
Ο = 0.97;
Ο = 0.1;
ΞΌ = -0.5 * Ο^2;
π = 5;

# Asset grid
π = 40;
πΜ = 50.0;
πΜ² = 0.0;
Ξ· = 0.4;

# Simulation
π = 50000;
π = 300;

# Computation general
max_it = 1000;
tol = 1.0E-6;

# Computation capital-labor ratio
max_it_KL = 100;
tol_KL = 1.0E-5;
step_KL = 0.005;
π0 = 1 / Ξ² - 1 - 0.001;
πΎπΏ0 = ((π0 + Ξ΄) / Ξ±)^(1 / (Ξ± - 1));

## Functions

# Utility function
function utility(π)
    if Ξ³ == 1.0
        π’ = log(π)
    elseif Ξ³ != 1.0
        π’ = (π^(1 - Ξ³) - 1) / (1 - Ξ³)
    end
    return π’
end;

# Derivative of utility function
function utility_derivative(π)
    π’ = π^(-Ξ³)
    return π’
end;

# Inverse of derivative of utility function
function utility_derivative_inverse(π’)
    π = π’^(-1 / Ξ³)
    return π
end;

# Interest rate
function r(πΎπΏ)
    π = Ξ± * π΄ * πΎπΏ^(Ξ± - 1) - Ξ΄
    return π
end;

# Wage
function w(πΎπΏ)
    π€ = (1 - Ξ±) * π΄ * πΎπΏ^Ξ±
    return π€
end;

# Linear interpolation
function lininterp(π±, π², π₯)
    # Find first index in vector π± below π₯
    π₯ = findlast(x -> x < π₯, π±)
    # Adjust for lower boundary
    if π₯ != nothing
        π₯ = π₯
    elseif π₯ == nothing
        π₯ = 1
    end
    # Find first index in vector π± below π₯
    π‘ = findfirst(x -> x > π₯, π±)
    # Adjust for upper boundary of grid set
    if π‘ != nothing
        π‘ = π‘
    elseif π‘ == nothing
        π‘ = length(π±)
    end
    # Interpolate
    if π₯ < π‘
        π¦ = π²[π₯] + (π₯ - π±[π₯]) * (π²[π‘] - π²[π₯]) / (π±[π‘] - π±[π₯])
    elseif π₯ == π‘
        π¦ = π²[π‘]
    end
    # Return output
    return π¦
end;

# Rouwenhorst
struct rouwenhorst
    # Define output
    π²::Vector{Float64} # Grid for gross efficiency units of labor
    π::Matrix{Float64} # Transition probability matrix
    π::Vector{Float64} # Stationary distribution
    # Define function
    function rouwenhorst(π, ΞΌ, Ο, Ο)
        # Transitition matrix parameter
        π = (1 + Ο) / 2
        # Width parameter
        π = sqrt((π - 1) * Ο^2 / (1 - Ο^2))
        # Grid for income (before normalization)
        π¦ = exp.(collect(range((ΞΌ - π), stop = (ΞΌ + π), length = π)))
        # Transition probability matrix for π=2
        π = [π (1-π); (1-π) π]
        # Two cases for π
        if π == 2
            π = π
        elseif π > 2
            for π = 1:(π-2)
                # (π-1) vector of zeros
                π = zeros(size(π, 2))
                # Update transititon probability matrix
                π =
                    (π * [π π; π' 0]) +
                    ((1 - π) * [π π; 0 π']) +
                    ((1 - π) * [π' 0; π π]) +
                    (π * [0 π'; π π])
            end
            # Ensure elements in each row sum to one
            π = Matrix(undef, π, π)
            π[1, :] = π[1, :]
            π[π, :] = π[π, :]
            for π = 2:(π-1)
                π[π, :] = π[π, :] ./ sum(π[π, :])
            end
        end
        # Stationary probability mass function
        π = (ones(π) ./ π)'
        for π = 1:1000
            π = π * (π^π)
        end
        # Convert into a column vector
        π = π'
        # Adjust grid income to normalize mean income to one
        π² = π¦ ./ sum(π¦ .* π)
        # Return output
        new(π², π, π)
    end
end;

## Discretize AR(1) Process for Efficiency Units of Labor

# Discretize using Rouwenhorst method
discretize = rouwenhorst(π, ΞΌ, Ο, Ο);

# Transition probability matrix
π = discretize.π;

# Stationary probability mass function
π = discretize.π;

# Recover grid for efficiency units of labor
π = discretize.π²;

## Asset Grid

π = collect(range(0, stop = 1, length = π));
π₯ = π .^ (1 / Ξ·);
π = πΜ² .+ (πΜ - πΜ²) .* π₯;

## Simulate efficiency units of labor

# Simulate i.i.d. uniform draws
π = rand(Uniform(0, 1), π, π);

# Initialize initial labor efficiency at lowest realization
π_sim = Matrix(undef, π, π);
for π = 1:π
    π_sim[π, 1] = π[1]
end;

# Simulate Markov chain
for π‘ = 2:π
    for π = 1:π
        # Locate last period's efficiency units of labor
        π§_lag = π_sim[π, (π‘-1)]
        # Identify index
        index = searchsortedfirst(π, π§_lag)
        # Select relevant row of transition matrix
        row = π[index, :]
        # Update this period's efficiency units of labor using transition probability matrix
        if π[π, π‘] < row[1]
            π_sim[π, π‘] = π[1]
        elseif row[1] < π[π, π‘] < sum(row[1:2])
            π_sim[π, π‘] = π[2]
        elseif sum(row[1:2]) < π[π, π‘] < sum(row[1:3])
            π_sim[π, π‘] = π[3]
        elseif sum(row[1:3]) < π[π, π‘] < sum(row[1:4])
            π_sim[π, π‘] = π[4]
        elseif sum(row[1:4]) < π[π, π‘] < sum(row[1:5])
            π_sim[π, π‘] = π[5]
        end
    end
end;

# Initialize assets
π_sim = zeros(π, π);

# Compute mean efficiency units of labor exogenously
πΏ = sum(π .* π);

## Capital Market

# Evaluate supply and demand for a given capital-labor ratio
struct capital
    π::Float64
    supply::Float64
    demand::Float64
    function capital(πΎπΏ)
        # Interest rate
        π = r(πΎπΏ)
        # Wage
        π€ = w(πΎπΏ)
        # Lump-sum transfer
        π = Ο * π€ * sum(π .* π)
        # Solve for consumption policy function using endogenous grid points
        π = π_guess
        iter = 0
        π_diff = Inf
        while iter <= max_it && π_diff > tol
            # Interpolating function
            function a(πβ², π§)
                # Initialize
                π¦π?_interp = Vector(undef, π)
                # Loop over next period's efficiency units of labor
                for π§β² = 1:π
                    # Interpolate consumption
                    π_interp = lininterp(π, π[:, π§β²], π[πβ²])
                    # Marginal utility
                    π¦π?_interp[π§β²] = utility_derivative(π_interp)
                end
                # Next period's expected marginal utility of consumption
                ππ¦π?π = sum(π¦π?_interp .* π[π§, :])
                # Compute RHS of Euler equation
                π¦π? = Ξ² * (1 + π) * ππ¦π?π
                # Consumption this period
                π = utility_derivative_inverse(π¦π?)
                # Assets this period
                π = (π + π[πβ²] - (1 - Ο) * π€ * π[π§] - π) / (1 + π)
                # Return output
                return π
            end
            # Evaluate assets this period as a function of asset's next period
            π1 = Matrix(undef, π, π)
            for π§ = 1:π
                for πβ² = 1:π
                    π1[πβ², π§] = a(πβ², π§)
                end
            end
            # Evaluate policy functions
            π_new = Matrix(undef, π, π)
            for π§ = 1:π
                for π = 1:π
                    # Borrowing constraint binds
                    if π[π] < π1[1, π§]
                        π[π, π§] = πΜ²
                        # Borrowing constraint does not bind
                    else
                        π[π, π§] = lininterp(π1[:, π§], π, π[π])
                    end
                    # Consumption policy function
                    π_new[π, π§] = (1 + π) * π[π] + (1 - Ο) * π€ * π[π§] + π - π[π, π§]
                end
            end
            # Compute difference
            π_diff = maximum(abs.(π_new - π))
            # Update consumption policy function
            π = π_new
            # Update iteration number
            iter = iter + 1
        end
        # Simulate policy functions
        π_sim = zeros(π, π)
        # Simulate assets
        π_sim = Matrix(undef, π, π)
        for π‘ = 2:π
            for π = 1:π
                # Recover last period's efficiency units of labor and its index
                π³ = π_sim[π, π‘-1]
                π§ = searchsortedfirst(π, π³)
                # Recover last period's asset value
                π = π_sim[π, π‘-1]
                # Interpolate consumption
                π_sim[π, π‘-1] = lininterp(π, π[:, π§], π)
                # Next period's optimal asset choice
                π_sim[π, π‘] = (1 + π) * π + (1 - Ο) * π€ * π³ + π - π_sim[π, π‘-1]
            end
        end
        # Capital-labor supply
        supply = mean(π_sim[:, π]) / πΏ
        # Capital-labor demand
        demand = πΎπΏ
        # Return output
        new(π, supply, demand)
    end
end;

# Evaluate endogenous variables as a function of interest rate
πΎπΏ_grid = collect(range(1.55, stop = 1.7, length = 5));
market = Vector(undef, length(πΎπΏ_grid));
eq = Matrix(undef, length(πΎπΏ_grid), 3);
for π in eachindex(πΎπΏ_grid)
    market[π] = capital(πΎπΏ_grid[π])
    eq[π, 1] = market[π].π
    eq[π, 2] = market[π].supply
    eq[π, 3] = market[π].demand
end;

# Plot supply and demand for capital
plot(eq[:, 2],
    eq[:, 1],
    xlabel = "Capital-Labor Ratio",
    ylabel = "Interest Rate",
    label = "Supply",
    color = :green,
    legend = :bottomright
)
plot!(eq[:, 3],
    eq[:, 1],
    xlabel = "Capital-Labor Ratio",
    ylabel = "Interest Rate",
    label = "Demand",
    color = :grey,
    legend = :bottomright
)
savefig("capital_market.png");

## Endogenous Grid Points

# Initialize consumption policy function
π_guess = Matrix(undef, π, π)
for π§ = 1:π
    for π = 1:π
        π_guess[π, π§] = π0 * π[π] + (1 - Ο) * π[π§]
    end
end;

# Set initial values
π = π_guess;
πΎπΏ = πΎπΏ0;
iter_KL = 0;
diff_KL = Inf;

# Initialize savings policy function
π = Matrix(undef, π, π);

# Iterate on the capital-labor ratio (NB: This is very slow, it's better to solve asset market directly)
while iter_KL <= max_it_KL && diff_KL > tol_KL
    # Interest rate
    π = r(πΎπΏ)
    # Wage
    π€ = w(πΎπΏ)
    # Lump-sum transfer
    π = Ο * π€ * sum(π .* π)
    # Solve for consumption policy function using endogenous grid points
    iter = 0
    π_diff = Inf
    while iter <= max_it && π_diff > tol
        # Interpolating function
        function a(πβ², π§)
            # Initialize
            π¦π?_interp = Vector(undef, π)
            # Loop over next period's efficiency units of labor
            for π§β² = 1:π
                # Interpolate consumption
                π_interp = lininterp(π, π[:, π§β²], π[πβ²])
                # Marginal utility
                π¦π?_interp[π§β²] = utility_derivative(π_interp)
            end
            # Next period's expected marginal utility of consumption
            ππ¦π?π = sum(π¦π?_interp .* π[π§, :])
            # Compute RHS of Euler equation
            π¦π? = Ξ² * (1 + π) * ππ¦π?π
            # Consumption this period
            π = utility_derivative_inverse(π¦π?)
            # Assets this period
            π = (π + π[πβ²] - (1 - Ο) * π€ * π[π§] - π) / (1 + π)
            # Return output
            return π
        end
        # Evaluate assets this period as a function of asset's next period
        π1 = Matrix(undef, π, π)
        for π§ = 1:π
            for πβ² = 1:π
                π1[πβ², π§] = a(πβ², π§)
            end
        end
        # Evaluate policy functions
        π_new = Matrix(undef, π, π)
        for π§ = 1:π
            for π = 1:π
                # Borrowing constraint binds
                if π[π] < π1[1, π§]
                    π[π, π§] = πΜ²
                    # Borrowing constraint does not bind
                else
                    π[π, π§] = lininterp(π1[:, π§], π, π[π])
                end
                # Consumption policy function
                π_new[π, π§] = (1 + π) * π[π] + (1 - Ο) * π€ * π[π§] + π - π[π, π§]
            end
        end
        # Compute difference
        π_diff = maximum(abs.(π_new - π))
        # Update consumption policy function
        π = π_new
        # Update iteration number
        iter = iter + 1
    end
    # Initialize assets to assets in terminal period from prior iteration
    π_sim[:, 1] = π_sim[:, π]
    # Simulate assets
    π_sim = Matrix(undef, π, π)
    for π‘ = 2:π
        for π = 1:π
            # Recover last period's efficiency units of labor and its index
            π³ = π_sim[π, π‘-1]
            π§ = searchsortedfirst(π, π³)
            # Recover last period's asset value
            π = π_sim[π, π‘-1]
            # Interpolate consumption
            π_sim[π, π‘-1] = lininterp(π, π[:, π§], π)
            # Next period's optimal asset choice
            π_sim[π, π‘] = (1 + π) * π + (1 - Ο) * π€ * π³ + π - π_sim[π, π‘-1]
        end
    end
    # Compute mean assets implied by stationary wealth distribution
    πΌπ = mean(π_sim[:, π])
    # New capital-labor ratio
    πΎπΏ_new = πΌπ / πΏ
    # Compute difference
    diff_KL = (πΎπΏ_new / πΎπΏ) - 1
    # Update capital-labor ratio using weighted average formula
    πΎπΏ = (1 - step_KL) * πΎπΏ + step_KL * πΎπΏ_new
    # Update iteration number
    iter_KL = iter_KL + 1
end;

## Results

# Equilibrium factor prices
π = r(πΎπΏ);
π€ = w(πΎπΏ);
π = Ο * π€ * sum(π .* π);
πΎπΏ_demand = (Ξ± / (1 - Ξ±)) * (π€ / (π + Ξ΄));
π_mean = mean(π_sim[:, π]);
π_median = quantile(π_sim[:, π], 0.5);
π_99 = quantile(π_sim[:, π], 0.99);
π_99_50 = π_99 / π_median;
π_zero = count(i -> i == πΜ², π_sim[:, π]) / length(π_sim[:, π]);

# Table with results
key = [L"r", L"w", L"T", L"K/L", "Mean wealth", "Median wealth", "99th percentile", "99th-50th ratio", "Fraction with zero wealth"];
table = TableCol(L"\tau=0.15", key, [π; π€; π; πΎπΏ_demand; π_mean; π_median; π_99; π_99_50; π_zero]);
to_tex(table) |> print

# Plot consumption policy function
plot(
    π,
    π[:, 1],
    xrange = (πΜ², πΜ),
    color = :blue,
    legend = :bottomright,
    label = "Lowest Income State",
    xlabel = "Assets",
    ylabel = "Consumption",
);
plot!(
    π,
    π[:, 5],
    xrange = (πΜ², πΜ),
    color = :red,
    legend = :bottomright,
    label = "Highest Income State",
    xlabel = "Assets",
    ylabel = "Consumption",
);
savefig("consumption_policy_function_income_state.png");

# Plot savings policy function
π_rate = Matrix(undef, π, π)
for π§ = 1:π
    for π = 1:π
        π_rate[π, π§] = π[π, π§] - π[π]
    end
end;
plot(
    π,
    π_rate[:, 1],
    xrange = (πΜ², πΜ),
    color = :blue,
    legend = :topright,
    label = "Lowest Income State",
    xlabel = "Assets",
    ylabel = "Savings",
);
plot!(
    π,
    π_rate[:, 5],
    xrange = (πΜ², πΜ),
    color = :red,
    legend = :topright,
    label = "Highest Income State",
    xlabel = "Assets",
    ylabel = "Savings",
);
savefig("savings_policy_function_income_state.png");

# Plot stationary wealth distribution
histogram(
    π_sim[:, π],
    xrange = (minimum(π_sim[:, π]), maximum(π_sim[:, π])),
    legend = false,
    color = :grey,
    xlabel = "Assets",
    ylabel = "Frequency",
);
savefig("wealth_distribution.png");
