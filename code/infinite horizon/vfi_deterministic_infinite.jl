################################################################################
#
#   PURPOSE: Infinite Horizon Deterministic Value Function Iteration
#   AUTHOR: Arshia Hashemi
#   EMAIL: arshiahashemi@uchicago.edu
#   DATE: Spring 2022
#
################################################################################

## Preamble

# Working directory
cd(
    "C:\\Users\\arshi\\Dropbox\\ArshiaHashemiRA\\ECMA 33620\\sessions\\session1\\",
);

# Packages
using Statistics, Random, LinearAlgebra, Plots

## Parameters

# Coefficient of relative risk aversion
γ = 2.0;

# Discount rate
β = 0.95;

# Net interest rate
𝑟 = 0.03;

# Gross interest rate
𝑅 = 1 + 𝑟;

# Labor income
𝑦 = 1.0;

# Asset lower bound
𝑎̲ = 0.0;

# Asset upper bound
𝑎̄ = 20.0;

# Dimension of asset grid
𝑁 = 1000;

# Asset grid
𝐴 = collect(range(𝑎̲, stop = 𝑎̄, length = 𝑁));

# Maximum number of iterations
max_it = 1000;

# Tolerance
tol = 1.0E-6;

## Functions

# Utility function
function utility(𝑐::Float64)
    if γ == 1
        𝑢 = log(𝑐)
    elseif γ != 1
        𝑢 = (𝑐^(1 - γ) - 1) / (1 - γ)
    end
    return 𝑢
end;

## Value Function Iteration

# Initial guess for value function
𝐕0 = Vector{Float64}(undef, 𝑁);
for 𝑎 = 1:𝑁
    𝐕0[𝑎] = utility(𝑟 * 𝐴[𝑎] + 𝑦) / (1 - β)
end;

# Initial values
𝐕 = 𝐕0;
iter = 0;
normdiff = Inf;

# Initialize matrices
𝒱 = Vector{Float64}(undef, 𝑁)
𝐀 = Vector{Float64}(undef, 𝑁);
𝐂 = Vector{Float64}(undef, 𝑁);
𝐒 = Vector{Float64}(undef, 𝑁);

# Iteration
while iter <= max_it && normdiff > tol
    # Loop over current period's state variable 𝑎
    for 𝑎 = 1:𝑁
        # Initialize matrices
        𝐶 = Vector{Float64}(undef, 𝑁)
        𝑉 = Vector{Float64}(undef, 𝑁)
        # Loop over next period's state variable 𝑎′
        for 𝑎′ = 1:𝑁
            # Candidate consumption
            𝐶[𝑎′] = 𝑅 * 𝐴[𝑎] + 𝑦 - 𝐴[𝑎′]
            # Candidate value function
            𝑉[𝑎′] = utility(max(𝐶[𝑎′], 1.0E-10)) + β * 𝐕[𝑎′]
        end
        # Index for optimal asset choice
        𝑎′ = argmax(𝑉)
        # Optimal asset choice
        𝐀[𝑎] = 𝐴[𝑎′]
        # Optimal consumption
        𝐂[𝑎] = 𝑅 * 𝐴[𝑎] + 𝑦 - 𝐀[𝑎]
        # Optimal savings
        𝐒[𝑎] = 𝐀[𝑎] - 𝐴[𝑎]
        # New value function
        𝒱[𝑎] = utility(𝐂[𝑎]) + β * 𝐕[𝑎′]
    end
    # Norm difference
    normdiff = norm(𝐕 - 𝒱)
    # Update values
    iter = iter + 1
    𝐕 = 𝒱
end;

## Policy Functions

# Consumption policy function
plot(
    𝐴,
    𝐂,
    xlabel = "Current Assets",
    ylabel = "Consumption",
    legend = false,
    color = :red,
    linestyle = :solid,
);
savefig("consumption.png");

# Savings policy function
plot(
    𝐴,
    𝐒,
    xlabel = "Current Assets",
    ylabel = "Savings",
    legend = false,
    color = :blue,
    linestyle = :solid,
);
savefig("savings.png");

## Monte Carlo Simulation

# Time periods
𝐓 = 500;

# Agents
𝐍 = 100;

# Set seed
Random.seed!(1234);

# Simulate U[0,1] values
𝑠 = rand(𝐍);

# Simulate initial asset values
𝐀_init = 𝑎̲ .+ (𝑠 .* 𝑎̄);

# Initialize matrices
𝐂_sim = Matrix{Float64}(undef, 𝐍, 𝐓);
𝐀_sim = Matrix{Float64}(undef, 𝐍, 𝐓 + 1);

# Set initial asset values
𝐀_sim[:, 1] = 𝐀_init;

# Simulate asset and consumption policy functions
for 𝑖 = 1:𝐍
    for 𝑡 = 1:𝐓
        # Asset index
        𝑎 = searchsortedfirst(𝐴, 𝐀_sim[𝑖, 𝑡])
        # Consumption
        𝐂_sim[𝑖, 𝑡] = 𝐂[𝑎]
        # Next period's asset choice
        𝐀_sim[𝑖, 𝑡+1] = 𝐀[𝑎]
    end
end;

# Means
𝐀_mean = dropdims(mean(𝐀_sim, dims = 1), dims = 1);
𝐂_mean = dropdims(mean(𝐂_sim, dims = 1), dims = 1);

# Asset dynamics
plot(
    𝐀_mean,
    seriestype = :line,
    color = :blue,
    xlabel = "Time Period",
    ylabel = "Mean Assets",
    legend = false,
);
savefig("asset_dynamics.png");

# Consumption dynamics
plot(
    𝐂_mean,
    seriestype = :line,
    color = :red,
    xlabel = "Time Period",
    ylabel = "Mean Consumption",
    legend = false,
);
savefig("consumption_dynamics.png");
