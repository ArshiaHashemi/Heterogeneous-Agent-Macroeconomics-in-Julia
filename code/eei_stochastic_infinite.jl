################################################################################
#
#   PURPOSE: Infinite Horizon Stochastic Euler Equation Iteration
#   AUTHOR: Arshia Hashemi
#   EMAIL: arshiahashemi@uchicago.edu
#   DATE: Spring 2022
#
################################################################################

## Preamble

# Working directory
cd(
    "C:\\Users\\arshi\\Dropbox\\ArshiaHashemiRA\\ECMA 33620\\assignments\\assignment2\\output\\",
);

# Packages
using Statistics, Random, LinearAlgebra, Plots, Roots, Distributions, TexTables

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

## Euler Equation Iteration

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
    # Initalize policy functions
    𝐂_new = Matrix(undef, 𝑁, 𝑛)
    # Interpolating Function
    function emuc(𝐜, 𝑥, 𝑦)
        # Compute next period's asset value given current period consumption 𝐜
        𝐚′ = 𝑋[𝑥, 𝑦] - 𝐜
        # Intialize vector with next period's interpolated consumption functions
        𝐦𝐮_interp = Vector(undef, 𝑛)
        # Loop over next period's labor income
        for 𝑦′ = 1:𝑛
            # Compute next period's cash on hand given net labor income 𝑦′
            𝐱′ = (1 + 𝑟) * 𝐚′ + 𝑌_net[𝑦′]
            # Find index of first grid point below 𝐱′
            𝐥 = findlast(x -> x < 𝐱′, 𝑋[:, 𝑦′])
            # Adjust for lower boundary of grid set
            if 𝐥 != nothing
                𝐥 = 𝐥
            elseif 𝐥 == nothing
                𝐥 = 1
            end
            # Find index of first grid point above 𝐱′
            𝐡 = findfirst(x -> x > 𝐱′, 𝑋[:, 𝑦′])
            # Adjust for upper boundary of grid set
            if 𝐡 != nothing
                𝐡 = 𝐡
            elseif 𝐡 == nothing
                𝐡 = length(𝑋[:, 𝑦′])
            end
            # Interpolate next period's consumption function
            if 𝐥 < 𝐡
                𝐂_interp =
                    𝐂[𝐥, 𝑦′] +
                    (𝐱′ - 𝑋[𝐥, 𝑦′]) * (𝐂[𝐡, 𝑦′] - 𝐂[𝐥, 𝑦′]) /
                    (𝑋[𝐡, 𝑦′] - 𝑋[𝐥, 𝑦′])
            elseif 𝐥 == 𝐡
                𝐂_interp = 𝐂[𝐡, 𝑦′]
            end
            # Evaluate marginal utility at interlopolated consumption
            𝐦𝐮_interp[𝑦′] = utility_derivative(𝐂_interp)
        end
        # Compute expected marginal utility next period
        𝐞𝐦𝐮𝐜 = sum(𝐦𝐮_interp .* 𝑃[𝑦, :])
        # Return output
        return 𝐞𝐦𝐮𝐜
    end
    # Loop over current cash on hand
    for 𝑥 = 1:𝑁
        # Loop over current labor income
        for 𝑦 = 1:𝑛
            # Consumption at borrowing constraint
            𝐜_lim = 𝑋[𝑥, 𝑦] - 𝑎̲
            # LHS of Euler equation if borrowing constraint binds
            𝐋 = utility_derivative(𝐜_lim)
            # RHS of Euler equation if borrowing constraint binds
            𝐑 = β * (1 + 𝑟) * emuc(𝐜_lim, 𝑥, 𝑦)
            # Generate indicator for whether borrowing constraint binds
            if 𝐋 >= 𝐑
                𝐈 = 1
            elseif 𝐋 < 𝐑
                𝐈 = 0
            end
            # Case 1: Borrowing constraint binds
            if 𝐈 == 1
                # Update savings policy function
                𝐒[𝑥, 𝑦] = 𝑎̲
                # Update consumption policy function
                𝐂_new[𝑥, 𝑦] = 𝑋[𝑥, 𝑦] - 𝐒[𝑥, 𝑦]
                # Case 2: Borrowing constraint does not bind
            elseif 𝐈 == 0
                # Define nonlinear equation to solve
                f(𝐜) = utility_derivative(𝐜) - β * (1 + 𝑟) * emuc(𝐜, 𝑥, 𝑦)
                # Solve nonlinear equation and update consumption policy function
                𝐂_new[𝑥, 𝑦] = find_zero(f, 𝐂[𝑥, 𝑦])
                # Update savings policy function
                𝐒[𝑥, 𝑦] = 𝑋[𝑥, 𝑦] - 𝐂_new[𝑥, 𝑦]
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
plot(𝑋[:, 1],
    𝐂[:, 1],
    xrange = (𝑎̲, 𝑎̅),
    color = :blue,
    legend = :bottomright,
    label = "Lowest Income State",
    xlabel = "Cash on Hand",
    ylabel = "Consumption",
);
plot!(𝑋[:, 5],
    𝐂[:, 5],
    xrange = (𝑎̲, 𝑎̅),
    color = :red,
    legend = :bottomright,
    label = "Highest Income State",
    xlabel = "Cash on Hand",
    ylabel = "Consumption",
);
savefig("consumption_policy_function_income_state.png");

# Compute savings rate
𝐒_rate = Matrix(undef, 𝑁, 𝑛)
for 𝑥 = 1:𝑁
    for 𝑦 = 1:𝑛
        𝐒_rate[𝑥, 𝑦] = 𝐒[𝑥, 𝑦] / 𝑋[𝑥, 𝑦]
    end
end;

# Savings rate by income state
plot(𝑋[:, 1],
    𝐒_rate[:, 1],
    xrange = (𝑎̲, 𝑎̅),
    yrange = (0.0, 1.0),
    color = :blue,
    legend = :bottomright,
    label = "Lowest Income State",
    xlabel = "Cash on Hand",
    ylabel = "Savings Rate",
);
plot!(𝑋[:, 5],
    𝐒_rate[:, 5],
    xrange = (𝑎̲, 𝑎̅),
    yrange = (0.0, 1.0),
    color = :red,
    legend = :bottomright,
    label = "Highest Income State",
    xlabel = "Cash on Hand",
    ylabel = "Savings Rate",
);
savefig("savings_policy_function_income_state.png");

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

# Interpolating function for consumption
function interpolation(𝐱, index)
    # Find index of first grid point below 𝐱
    𝐥 = findlast(x -> x < 𝐱, 𝑋[:, index])
    # Adjust for lower boundary of grid set
    if 𝐥 != nothing
        𝐥 = 𝐥
    elseif 𝐥 == nothing
        𝐥 = 1
    end
    # Find index of first grid point above 𝐱
    𝐡 = findfirst(x -> x > 𝐱, 𝑋[:, index])
    # Adjust for upper boundary of grid set
    if 𝐡 != nothing
        𝐡 = 𝐡
    elseif 𝐡 == nothing
        𝐡 = length(𝑋[:, index])
    end
    # Interpolate last period's consumption function
    if 𝐥 < 𝐡
        𝐂_interp =
            𝐂[𝐥, index] +
            (𝐱 - 𝑋[𝐥, index]) * (𝐂[𝐡, index] - 𝐂[𝐥, index]) /
            (𝑋[𝐡, index] - 𝑋[𝐥, index])
    elseif 𝐥 == 𝐡
        𝐂_interp = 𝐂[𝐡, index]
    end
    # Return output
    return 𝐂_interp
end;

# Simulate policy functions
for 𝑡 = 2:𝐓
    for 𝑖 = 1:𝐍
        # Recover last period's net labor income and its index
        𝐲 = 𝐘_sim[𝑖, 𝑡 - 1]
        index = searchsortedfirst(𝑌_net, 𝐲)
        # Recover last period's asset value
        𝐚 = 𝐀_sim[𝑖, 𝑡 - 1]
        # Compute last period's cash on hand
        𝐱 = (1 + 𝑟) * 𝐚 + 𝐲
        # Last period's optimal consumption
        𝐂_sim[𝑖, 𝑡 - 1] = interpolation(𝐱, index)
        # Current period's optimal asset choice
        𝐀_sim[𝑖, 𝑡] = 𝐱 - 𝐂_sim[𝑖, 𝑡 - 1]
    end
end;

## Plot Simulations

# Plot net labor income distribution in terminal period
histogram(𝐘_sim[:, 𝐓],
    legend = false,
    color = :blue,
    xrange = (0.5, 1.8),
    xlabel = "Net Labor Income",
    ylabel = "Frequency",
);
savefig("net_labor_income_distribution.png");

# Plot asset distribution in terminal period
histogram(𝐀_sim[:, 𝐓],
    legend = false,
    color = :grey,
    xrange = (-1.0, 50.0),
    xlabel = "Assets",
    ylabel = "Frequency",
);
savefig("asset_distribution.png");

# Asymptotic distributions
𝐘_dist = 𝐘_sim[:, 𝐓];
𝐀_dist = 𝐀_sim[:, 𝐓];

# Statistics for wealth
𝐀_zero = count(i -> i == 0.0, 𝐀_dist) / length(𝐀_dist);
𝐀_mean = mean(𝐀_dist);
𝐀_median = median(𝐀_dist);
𝐀_90 = quantile(𝐀_dist, 0.9);
𝐀_99 = quantile(𝐀_dist, 0.99);

# Table with wealth statistics
key = ["Fraction with zero assets", "Mean", "Median", "90th percentile", "99th percentile"];
inequality = TableCol("High Income Tax", key, [𝐀_zero; 𝐀_mean; 𝐀_median; 𝐀_90; 𝐀_99]);
to_tex(inequality) |> print
