################################################################################
#
#   PURPOSE: Infinite Horizon Dynamic Programming with Endogenous Grid Points
#   AUTHOR: Arshia Hashemi
#   EMAIL: arshiahashemi@uchicago.edu
#   FIRST VERSION: 04/05/2022
#   THIS VERSION: 05/01/2022
#
################################################################################

## Preamble

# Working directory
cd(
    "C:\\Users\\arshi\\Dropbox\\ArshiaHashemiRA\\ECMA 33620\\assignments\\assignment2\\",
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

# Risk aversion
γ = 1.5;

# Discout rate
β = 0.975;

# Net interest rate
𝑟 = 0.02;

# Labor income tax
τ = 0.3;

# Asset lower bound
𝑎̲ = 0.0;

# Asset upper bound
𝑎̅ = 40.0;

# Dimension of asset grid
𝑁 = 50;

# Nonlinearity of asset grid
α = 0.4;

# Number of discrete income states
𝑛 = 5;

# Persistence of log income process
ρ = 0.97;

# Standard deviation of innovation shock to log income
σ = 0.1;

# Mean of innovation shock to log income
μ = -0.5 * σ^2;

# Number of households for simulation
𝐍 = 100000;

# Number of time periods for simulation
𝐓 = 300;

# Maximum number of iterations
max_it = 1000;

# Tolerance
tol = 1.0E-6;

## Functions

# Utility function
function utility(𝑐)
    if γ == 1.0
        𝑢 = log(𝑐)
    elseif γ != 1.0
        𝑢 = (𝑐^(1 - γ) - 1) / (1 - γ)
    end
    return 𝑢
end;

# Derivative of utility function
function utility_derivative(𝑐)
    𝑢 = 𝑐^(-γ)
    return 𝑢
end;

# Inverse of derivative of utility function
function utility_derivative_inverse(𝑢)
    𝑐 = 𝑢^(-1 / γ)
    return 𝑐
end;

# Linear interpolation
function lininterp(𝐱, 𝐲, 𝑥)
    # Find first index in vector 𝐱 below 𝑥
    𝐥 = findlast(x -> x < 𝑥, 𝐱)
    # Adjust for lower boundary
    if 𝐥 != nothing
        𝐥 = 𝐥
    elseif 𝐥 == nothing
        𝐥 = 1
    end
    # Find first index in vector 𝐱 below 𝑥
    𝐡 = findfirst(x -> x > 𝑥, 𝐱)
    # Adjust for upper boundary of grid set
    if 𝐡 != nothing
        𝐡 = 𝐡
    elseif 𝐡 == nothing
        𝐡 = length(𝐱)
    end
    # Interpolate
    if 𝐥 < 𝐡
        𝑦 = 𝐲[𝐥] + (𝑥 - 𝐱[𝐥]) * (𝐲[𝐡] - 𝐲[𝐥]) / (𝐱[𝐡] - 𝐱[𝐥])
    elseif 𝐥 == 𝐡
        𝑦 = 𝐲[𝐡]
    end
    # Return output
    return 𝑦
end;

# Rouwenhorst
struct rouwenhorst
    # Define output
    𝐲::Vector{Float64} # Grid for gross labor income
    𝐏::Matrix{Float64} # Transition probability matrix
    𝛑::Vector{Float64} # Stationary distribution
    # Define function
    function rouwenhorst(𝑛, μ, ρ, σ)
        # Transitition matrix parameter
        𝑝 = (1 + ρ) / 2
        # Width parameter
        𝜓 = sqrt((𝑛 - 1) * σ^2 / (1 - ρ^2))
        # Grid for income (before normalization)
        𝑦 = exp.(collect(range((μ - 𝜓), stop = (μ + 𝜓), length = 𝑛)))
        # Transition probability matrix for 𝑛=2
        𝑃 = [𝑝 (1-𝑝); (1-𝑝) 𝑝]
        # Two cases for 𝑛
        if 𝑛 == 2
            𝐏 = 𝑃
        elseif 𝑛 > 2
            for 𝑖 = 1:(𝑛-2)
                # (𝑛-1) vector of zeros
                𝟎 = zeros(size(𝑃, 2))
                # Update transititon probability matrix
                𝑃 =
                    (𝑝 * [𝑃 𝟎; 𝟎' 0]) +
                    ((1 - 𝑝) * [𝟎 𝑃; 0 𝟎']) +
                    ((1 - 𝑝) * [𝟎' 0; 𝑃 𝟎]) +
                    (𝑝 * [0 𝟎'; 𝟎 𝑃])
            end
            # Ensure elements in each row sum to one
            𝐏 = Matrix(undef, 𝑛, 𝑛)
            𝐏[1, :] = 𝑃[1, :]
            𝐏[𝑛, :] = 𝑃[𝑛, :]
            for 𝑟 = 2:(𝑛-1)
                𝐏[𝑟, :] = 𝑃[𝑟, :] ./ sum(𝑃[𝑟, :])
            end
        end
        # Stationary probability mass function
        𝛑 = (ones(𝑛) ./ 𝑛)'
        for 𝑖 = 1:1000
            𝛑 = 𝛑 * (𝐏^𝑖)
        end
        # Convert into a column vector
        𝛑 = 𝛑'
        # Adjust grid income to normalize mean income to one
        𝐲 = 𝑦 ./ sum(𝑦 .* 𝛑)
        # Return output
        new(𝐲, 𝐏, 𝛑)
    end
end;

## Discretize AR(1) Process for Labor Income

# Discretize using Rouwenhorst method
discretize = rouwenhorst(𝑛, μ, ρ, σ);

# Transition probability matrix
𝑃 = discretize.𝐏;

# Stationary probability mass function
𝜋 = discretize.𝛑;

# Recover gross labor income grid
𝑌_gross = discretize.𝐲;

# Lump sum transfer clears government's balanced budget
𝑇 = τ * sum(𝑌_gross .* 𝜋);

# Construct net labor income grid after taxes and transfers
𝑌_net = (1 - τ) .* 𝑌_gross .+ 𝑇;

## Asset Grid

# Construct cash on hand grid for each discrete net income level
𝑋 = Matrix(undef, 𝑁, 𝑛);
for 𝑦 = 1:𝑛
    # Equispaced grid on [0,1]
    𝑧 = collect(range(0, stop = 1, length = 𝑁))
    # Add nonlinearity
    𝑥 = 𝑧 .^ (1 / α)
    # Construct grid point
    𝑋[:, 𝑦] = 𝑎̲ .+ (𝑎̅ - 𝑎̲) .* 𝑥 .+ 𝑌_net[𝑦]
end;

# Savings grid
𝑧 = collect(range(0, stop = 1, length = 𝑁));
𝑥 = 𝑧 .^ (1 / α);
𝑆 = 𝑎̲ .+ (𝑎̅ - 𝑎̲) .* 𝑥;

## Endogenous Grid Points Method

# Initialize policy functions
𝐂_guess = Matrix(undef, 𝑁, 𝑛);
for 𝑦 = 1:𝑛
    𝐂_guess[:, 𝑦] = 𝑟 .* 𝑋[:, 𝑦]
end;

# Initial values
𝐂 = 𝐂_guess;
iter = 0;
𝐂_diff = Inf;

# Initialize savings policy function
𝐒 = Matrix(undef, 𝑁, 𝑛);

# Iteration
while iter <= max_it && 𝐂_diff > tol
    # Initalize policy function for current iteration
    𝐂_new = Matrix(undef, 𝑁, 𝑛)
    # Interpolating function
    function x(𝑠, 𝑦)
        # Initialize
        𝐦𝐮_interp = Vector(undef, 𝑛)
        # Loop over next period's labor income
        for 𝑦′ = 1:𝑛
            # Law of motion for cash on hand next period
            𝐱′ = (1 + 𝑟) * 𝑆[𝑠] + 𝑌_net[𝑦′]
            # Interpolate next period's consumption
            𝐂_interp = lininterp(𝑋[:, 𝑦′], 𝐂[:, 𝑦′], 𝐱′)
            # Evaluate marginal utility at interlopolated consumption
            𝐦𝐮_interp[𝑦′] = utility_derivative(𝐂_interp)
        end
        # Compute next period's expected marginal utility of consumption
        𝐞𝐦𝐮𝐜 = sum(𝐦𝐮_interp .* 𝑃[𝑦, :])
        # Compute RHS of Euler equation
        𝐦𝐮𝐜 = β * (1 + 𝑟) * 𝐞𝐦𝐮𝐜
        # Consumption as a function of current period's savings and labor income
        𝐜 = utility_derivative_inverse(𝐦𝐮𝐜)
        # Cash on hand as a function of current period's savings and labor income
        𝐱 = 𝑆[𝑠] + 𝐜
        # Return output
        return 𝐱
    end
    # Compute cash on hand as a function of current period's savings and labor income
    𝐗 = Matrix(undef, 𝑁, 𝑛)
    # Loop over current labor income
    for 𝑦 = 1:𝑛
        # Loop over savings
        for 𝑠 = 1:𝑁
            # Cash on hand this period
            𝐗[𝑠, 𝑦] = x(𝑠, 𝑦)
        end
    end
    # Loop over current period's labor income
    for 𝑦 = 1:𝑛
        # Loop over current period's cash on hand
        for 𝑥 = 1:𝑁
            # Borrowing constraint binds
            if 𝑋[𝑥, 𝑦] < 𝐗[1, 𝑦]
                𝐒[𝑥, 𝑦] = 𝑎̲
                𝐂_new[𝑥, 𝑦] = 𝑋[𝑥, 𝑦] - 𝐒[𝑥, 𝑦]
                # Borrowing constraint does not bind
            else
                # Cash on hand
                𝐱 = 𝑋[𝑥, 𝑦]
                # Interpolate savings policy function
                𝐒[𝑥, 𝑦] = lininterp(𝐗[:, 𝑦], 𝑆, 𝐱)
                # Recover consumption
                𝐂_new[𝑥, 𝑦] = 𝑋[𝑥, 𝑦] - 𝐒[𝑥, 𝑦]
            end
        end
    end
    # Norm difference
    𝐂_diff = maximum(abs.(𝐂_new - 𝐂))
    # Update values
    iter = iter + 1
    𝐂 = 𝐂_new
end;

## Plot Policy Functions

# Consumption policy function by income state
plot(
    𝑋[:, 1],
    𝐂[:, 1],
    xrange = (𝑎̲, 𝑎̅),
    color = :blue,
    legend = :bottomright,
    label = "Lowest Income State",
    xlabel = "Cash on Hand",
    ylabel = "Consumption",
);
plot!(
    𝑋[:, 5],
    𝐂[:, 5],
    xrange = (𝑎̲, 𝑎̅),
    color = :red,
    legend = :bottomright,
    label = "Highest Income State",
    xlabel = "Cash on Hand",
    ylabel = "Consumption",
);
savefig("output\\consumption_policy_function_income_state.png");

# Compute savings rate
𝐒_rate = Matrix(undef, 𝑁, 𝑛)
for 𝑥 = 1:𝑁
    for 𝑦 = 1:𝑛
        𝐒_rate[𝑥, 𝑦] = 𝐒[𝑥, 𝑦] / 𝑋[𝑥, 𝑦]
    end
end;

# Savings rate by income state
plot(
    𝑋[:, 1],
    𝐒_rate[:, 1],
    xrange = (𝑎̲, 𝑎̅),
    yrange = (0.0, 1.0),
    color = :blue,
    legend = :bottomright,
    label = "Lowest Income State",
    xlabel = "Cash on Hand",
    ylabel = "Savings Rate",
);
plot!(
    𝑋[:, 5],
    𝐒_rate[:, 5],
    xrange = (𝑎̲, 𝑎̅),
    yrange = (0.0, 1.0),
    color = :red,
    legend = :bottomright,
    label = "Highest Income State",
    xlabel = "Cash on Hand",
    ylabel = "Savings Rate",
);
savefig("output\\savings_policy_function_income_state.png");

## Simulation

# Set seed
Random.seed!(1234);

# Initalize state variables and policy functions
𝐘_sim = Matrix(undef, 𝐍, 𝐓);
𝐀_sim = Matrix(undef, 𝐍, 𝐓);
𝐂_sim = Matrix(undef, 𝐍, 𝐓);

# Simulate i.i.d. uniform draws
𝐔 = rand(Uniform(0, 1), 𝐍, 𝐓);

# Initialize state variable values in initial period
for 𝑖 = 1:𝐍
    𝐘_sim[𝑖, 1] = 𝑌_net[1]
    𝐀_sim[𝑖, 1] = 0.0
end;

# Simulate Markov chain
for 𝑡 = 2:𝐓
    for 𝑖 = 1:𝐍
        # Locate last period's labor income
        𝑦_lag = 𝐘_sim[𝑖, (𝑡-1)]
        # Identify index
        index = searchsortedfirst(𝑌_net, 𝑦_lag)
        # Select relevant row of transition matrix
        row = 𝑃[index, :]
        # Update this period's labor income using transition probability matrix
        if 𝐔[𝑖, 𝑡] < row[1]
            𝐘_sim[𝑖, 𝑡] = 𝑌_net[1]
        elseif row[1] < 𝐔[𝑖, 𝑡] < sum(row[1:2])
            𝐘_sim[𝑖, 𝑡] = 𝑌_net[2]
        elseif sum(row[1:2]) < 𝐔[𝑖, 𝑡] < sum(row[1:3])
            𝐘_sim[𝑖, 𝑡] = 𝑌_net[3]
        elseif sum(row[1:3]) < 𝐔[𝑖, 𝑡] < sum(row[1:4])
            𝐘_sim[𝑖, 𝑡] = 𝑌_net[4]
        elseif sum(row[1:4]) < 𝐔[𝑖, 𝑡] < sum(row[1:5])
            𝐘_sim[𝑖, 𝑡] = 𝑌_net[5]
        end
    end
end;

# Simulate consumption and asset policy functions
for 𝑡 = 2:𝐓
    for 𝑖 = 1:𝐍
        # Recover last period's net labor income and its index
        𝐲 = 𝐘_sim[𝑖, 𝑡-1]
        index = searchsortedfirst(𝑌_net, 𝐲)
        # Recover last period's asset value
        𝐚 = 𝐀_sim[𝑖, 𝑡-1]
        # Compute last period's cash on hand
        𝐱 = (1 + 𝑟) * 𝐚 + 𝐲
        # Interpolate optimal consumption
        𝐂_sim[𝑖, 𝑡-1] = lininterp(𝑋[:, index], 𝐂[:, index], 𝐱)
        # Current period's optimal asset choice
        𝐀_sim[𝑖, 𝑡] = 𝐱 - 𝐂_sim[𝑖, 𝑡-1]
    end
end;

## Plot Simulations

# Stionary distributions
𝐘_dist = 𝐘_sim[:, 𝐓];
𝐀_dist = 𝐀_sim[:, 𝐓];

# Plot net labor income distribution in terminal period
histogram(
    𝐘_dist,
    legend = false,
    color = :blue,
    xlabel = "Net Labor Income",
    ylabel = "Frequency",
);
savefig("output\\net_labor_income_distribution.png");

# Plot asset distribution in terminal period
histogram(
    𝐀_dist,
    legend = false,
    color = :grey,
    xrange = (minimum(𝐀_dist), maximum(𝐀_dist)/2),
    xlabel = "Assets",
    ylabel = "Frequency",
);
savefig("output\\wealth_distribution.png");

# Statistics of wealth distribution
𝐀_zero = count(i -> i == 0.0, 𝐀_dist) / length(𝐀_dist);
𝐀_mean = mean(𝐀_dist);
𝐀_median = median(𝐀_dist);
𝐀_90  = quantile(𝐀_dist, 0.9);
𝐀_99 = quantile(𝐀_dist, 0.99);
A_99_50 = 𝐀_99 / 𝐀_median;

# Table with wealth statistics
key = [
    "Fraction with zero assets",
    "Mean",
    "Median",
    "90th percentile",
    "99th percentile",
    "99th-50th ratio"
];
inequality =
    TableCol(L"\tau=0.15", key, [𝐀_zero; 𝐀_mean; 𝐀_median; 𝐀_90; 𝐀_99; A_99_50]);
to_tex(inequality) |> print
