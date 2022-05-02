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
γ = 1.5;
β = 0.95;

# efficiency units of labor tax
τ = 0.15;

# Production
𝐴 = 1.0;
α = 1 / 3;
δ = 0.2;

# Effiency units of labor process
ρ = 0.97;
σ = 0.1;
μ = -0.5 * σ^2;
𝑛 = 5;

# Asset grid
𝑁 = 40;
𝑎̅ = 50.0;
𝑎̲ = 0.0;
η = 0.4;

# Simulation
𝐍 = 50000;
𝐓 = 300;

# Computation general
max_it = 1000;
tol = 1.0E-6;

# Computation capital-labor ratio
max_it_KL = 100;
tol_KL = 1.0E-5;
step_KL = 0.005;
𝑟0 = 1 / β - 1 - 0.001;
𝐾𝐿0 = ((𝑟0 + δ) / α)^(1 / (α - 1));

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

# Interest rate
function r(𝐾𝐿)
    𝑟 = α * 𝐴 * 𝐾𝐿^(α - 1) - δ
    return 𝑟
end;

# Wage
function w(𝐾𝐿)
    𝑤 = (1 - α) * 𝐴 * 𝐾𝐿^α
    return 𝑤
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
    𝐲::Vector{Float64} # Grid for gross efficiency units of labor
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

## Discretize AR(1) Process for Efficiency Units of Labor

# Discretize using Rouwenhorst method
discretize = rouwenhorst(𝑛, μ, ρ, σ);

# Transition probability matrix
𝑃 = discretize.𝐏;

# Stationary probability mass function
𝜋 = discretize.𝛑;

# Recover grid for efficiency units of labor
𝐙 = discretize.𝐲;

## Asset Grid

𝑔 = collect(range(0, stop = 1, length = 𝑁));
𝑥 = 𝑔 .^ (1 / η);
𝐀 = 𝑎̲ .+ (𝑎̅ - 𝑎̲) .* 𝑥;

## Simulate efficiency units of labor

# Simulate i.i.d. uniform draws
𝐔 = rand(Uniform(0, 1), 𝐍, 𝐓);

# Initialize initial labor efficiency at lowest realization
𝐙_sim = Matrix(undef, 𝐍, 𝐓);
for 𝑖 = 1:𝐍
    𝐙_sim[𝑖, 1] = 𝐙[1]
end;

# Simulate Markov chain
for 𝑡 = 2:𝐓
    for 𝑖 = 1:𝐍
        # Locate last period's efficiency units of labor
        𝑧_lag = 𝐙_sim[𝑖, (𝑡-1)]
        # Identify index
        index = searchsortedfirst(𝐙, 𝑧_lag)
        # Select relevant row of transition matrix
        row = 𝑃[index, :]
        # Update this period's efficiency units of labor using transition probability matrix
        if 𝐔[𝑖, 𝑡] < row[1]
            𝐙_sim[𝑖, 𝑡] = 𝐙[1]
        elseif row[1] < 𝐔[𝑖, 𝑡] < sum(row[1:2])
            𝐙_sim[𝑖, 𝑡] = 𝐙[2]
        elseif sum(row[1:2]) < 𝐔[𝑖, 𝑡] < sum(row[1:3])
            𝐙_sim[𝑖, 𝑡] = 𝐙[3]
        elseif sum(row[1:3]) < 𝐔[𝑖, 𝑡] < sum(row[1:4])
            𝐙_sim[𝑖, 𝑡] = 𝐙[4]
        elseif sum(row[1:4]) < 𝐔[𝑖, 𝑡] < sum(row[1:5])
            𝐙_sim[𝑖, 𝑡] = 𝐙[5]
        end
    end
end;

# Initialize assets
𝐀_sim = zeros(𝐍, 𝐓);

# Compute mean efficiency units of labor exogenously
𝐿 = sum(𝐙 .* 𝜋);

## Capital Market

# Evaluate supply and demand for a given capital-labor ratio
struct capital
    𝑟::Float64
    supply::Float64
    demand::Float64
    function capital(𝐾𝐿)
        # Interest rate
        𝑟 = r(𝐾𝐿)
        # Wage
        𝑤 = w(𝐾𝐿)
        # Lump-sum transfer
        𝑇 = τ * 𝑤 * sum(𝐙 .* 𝜋)
        # Solve for consumption policy function using endogenous grid points
        𝐂 = 𝐂_guess
        iter = 0
        𝐂_diff = Inf
        while iter <= max_it && 𝐂_diff > tol
            # Interpolating function
            function a(𝑎′, 𝑧)
                # Initialize
                𝐦𝐮_interp = Vector(undef, 𝑛)
                # Loop over next period's efficiency units of labor
                for 𝑧′ = 1:𝑛
                    # Interpolate consumption
                    𝐜_interp = lininterp(𝐀, 𝐂[:, 𝑧′], 𝐀[𝑎′])
                    # Marginal utility
                    𝐦𝐮_interp[𝑧′] = utility_derivative(𝐜_interp)
                end
                # Next period's expected marginal utility of consumption
                𝐞𝐦𝐮𝐜 = sum(𝐦𝐮_interp .* 𝑃[𝑧, :])
                # Compute RHS of Euler equation
                𝐦𝐮 = β * (1 + 𝑟) * 𝐞𝐦𝐮𝐜
                # Consumption this period
                𝐜 = utility_derivative_inverse(𝐦𝐮)
                # Assets this period
                𝐚 = (𝐜 + 𝐀[𝑎′] - (1 - τ) * 𝑤 * 𝐙[𝑧] - 𝑇) / (1 + 𝑟)
                # Return output
                return 𝐚
            end
            # Evaluate assets this period as a function of asset's next period
            𝐀1 = Matrix(undef, 𝑁, 𝑛)
            for 𝑧 = 1:𝑛
                for 𝑎′ = 1:𝑁
                    𝐀1[𝑎′, 𝑧] = a(𝑎′, 𝑧)
                end
            end
            # Evaluate policy functions
            𝐂_new = Matrix(undef, 𝑁, 𝑛)
            for 𝑧 = 1:𝑛
                for 𝑎 = 1:𝑁
                    # Borrowing constraint binds
                    if 𝐀[𝑎] < 𝐀1[1, 𝑧]
                        𝐒[𝑎, 𝑧] = 𝑎̲
                        # Borrowing constraint does not bind
                    else
                        𝐒[𝑎, 𝑧] = lininterp(𝐀1[:, 𝑧], 𝐀, 𝐀[𝑎])
                    end
                    # Consumption policy function
                    𝐂_new[𝑎, 𝑧] = (1 + 𝑟) * 𝐀[𝑎] + (1 - τ) * 𝑤 * 𝐙[𝑧] + 𝑇 - 𝐒[𝑎, 𝑧]
                end
            end
            # Compute difference
            𝐂_diff = maximum(abs.(𝐂_new - 𝐂))
            # Update consumption policy function
            𝐂 = 𝐂_new
            # Update iteration number
            iter = iter + 1
        end
        # Simulate policy functions
        𝐀_sim = zeros(𝐍, 𝐓)
        # Simulate assets
        𝐂_sim = Matrix(undef, 𝐍, 𝐓)
        for 𝑡 = 2:𝐓
            for 𝑖 = 1:𝐍
                # Recover last period's efficiency units of labor and its index
                𝐳 = 𝐙_sim[𝑖, 𝑡-1]
                𝑧 = searchsortedfirst(𝐙, 𝐳)
                # Recover last period's asset value
                𝐚 = 𝐀_sim[𝑖, 𝑡-1]
                # Interpolate consumption
                𝐂_sim[𝑖, 𝑡-1] = lininterp(𝐀, 𝐂[:, 𝑧], 𝐚)
                # Next period's optimal asset choice
                𝐀_sim[𝑖, 𝑡] = (1 + 𝑟) * 𝐚 + (1 - τ) * 𝑤 * 𝐳 + 𝑇 - 𝐂_sim[𝑖, 𝑡-1]
            end
        end
        # Capital-labor supply
        supply = mean(𝐀_sim[:, 𝐓]) / 𝐿
        # Capital-labor demand
        demand = 𝐾𝐿
        # Return output
        new(𝑟, supply, demand)
    end
end;

# Evaluate endogenous variables as a function of interest rate
𝐾𝐿_grid = collect(range(1.55, stop = 1.7, length = 5));
market = Vector(undef, length(𝐾𝐿_grid));
eq = Matrix(undef, length(𝐾𝐿_grid), 3);
for 𝑖 in eachindex(𝐾𝐿_grid)
    market[𝑖] = capital(𝐾𝐿_grid[𝑖])
    eq[𝑖, 1] = market[𝑖].𝑟
    eq[𝑖, 2] = market[𝑖].supply
    eq[𝑖, 3] = market[𝑖].demand
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
𝐂_guess = Matrix(undef, 𝑁, 𝑛)
for 𝑧 = 1:𝑛
    for 𝑎 = 1:𝑁
        𝐂_guess[𝑎, 𝑧] = 𝑟0 * 𝐀[𝑎] + (1 - τ) * 𝐙[𝑧]
    end
end;

# Set initial values
𝐂 = 𝐂_guess;
𝐾𝐿 = 𝐾𝐿0;
iter_KL = 0;
diff_KL = Inf;

# Initialize savings policy function
𝐒 = Matrix(undef, 𝑁, 𝑛);

# Iterate on the capital-labor ratio
while iter_KL <= max_it_KL && diff_KL > tol_KL
    # Interest rate
    𝑟 = r(𝐾𝐿)
    # Wage
    𝑤 = w(𝐾𝐿)
    # Lump-sum transfer
    𝑇 = τ * 𝑤 * sum(𝐙 .* 𝜋)
    # Solve for consumption policy function using endogenous grid points
    iter = 0
    𝐂_diff = Inf
    while iter <= max_it && 𝐂_diff > tol
        # Interpolating function
        function a(𝑎′, 𝑧)
            # Initialize
            𝐦𝐮_interp = Vector(undef, 𝑛)
            # Loop over next period's efficiency units of labor
            for 𝑧′ = 1:𝑛
                # Interpolate consumption
                𝐜_interp = lininterp(𝐀, 𝐂[:, 𝑧′], 𝐀[𝑎′])
                # Marginal utility
                𝐦𝐮_interp[𝑧′] = utility_derivative(𝐜_interp)
            end
            # Next period's expected marginal utility of consumption
            𝐞𝐦𝐮𝐜 = sum(𝐦𝐮_interp .* 𝑃[𝑧, :])
            # Compute RHS of Euler equation
            𝐦𝐮 = β * (1 + 𝑟) * 𝐞𝐦𝐮𝐜
            # Consumption this period
            𝐜 = utility_derivative_inverse(𝐦𝐮)
            # Assets this period
            𝐚 = (𝐜 + 𝐀[𝑎′] - (1 - τ) * 𝑤 * 𝐙[𝑧] - 𝑇) / (1 + 𝑟)
            # Return output
            return 𝐚
        end
        # Evaluate assets this period as a function of asset's next period
        𝐀1 = Matrix(undef, 𝑁, 𝑛)
        for 𝑧 = 1:𝑛
            for 𝑎′ = 1:𝑁
                𝐀1[𝑎′, 𝑧] = a(𝑎′, 𝑧)
            end
        end
        # Evaluate policy functions
        𝐂_new = Matrix(undef, 𝑁, 𝑛)
        for 𝑧 = 1:𝑛
            for 𝑎 = 1:𝑁
                # Borrowing constraint binds
                if 𝐀[𝑎] < 𝐀1[1, 𝑧]
                    𝐒[𝑎, 𝑧] = 𝑎̲
                    # Borrowing constraint does not bind
                else
                    𝐒[𝑎, 𝑧] = lininterp(𝐀1[:, 𝑧], 𝐀, 𝐀[𝑎])
                end
                # Consumption policy function
                𝐂_new[𝑎, 𝑧] = (1 + 𝑟) * 𝐀[𝑎] + (1 - τ) * 𝑤 * 𝐙[𝑧] + 𝑇 - 𝐒[𝑎, 𝑧]
            end
        end
        # Compute difference
        𝐂_diff = maximum(abs.(𝐂_new - 𝐂))
        # Update consumption policy function
        𝐂 = 𝐂_new
        # Update iteration number
        iter = iter + 1
    end
    # Initialize assets to assets in terminal period from prior iteration
    𝐀_sim[:, 1] = 𝐀_sim[:, 𝐓]
    # Simulate assets
    𝐂_sim = Matrix(undef, 𝐍, 𝐓)
    for 𝑡 = 2:𝐓
        for 𝑖 = 1:𝐍
            # Recover last period's efficiency units of labor and its index
            𝐳 = 𝐙_sim[𝑖, 𝑡-1]
            𝑧 = searchsortedfirst(𝐙, 𝐳)
            # Recover last period's asset value
            𝐚 = 𝐀_sim[𝑖, 𝑡-1]
            # Interpolate consumption
            𝐂_sim[𝑖, 𝑡-1] = lininterp(𝐀, 𝐂[:, 𝑧], 𝐚)
            # Next period's optimal asset choice
            𝐀_sim[𝑖, 𝑡] = (1 + 𝑟) * 𝐚 + (1 - τ) * 𝑤 * 𝐳 + 𝑇 - 𝐂_sim[𝑖, 𝑡-1]
        end
    end
    # Compute mean assets implied by stationary wealth distribution
    𝔼𝐀 = mean(𝐀_sim[:, 𝐓])
    # New capital-labor ratio
    𝐾𝐿_new = 𝔼𝐀 / 𝐿
    # Compute difference
    diff_KL = (𝐾𝐿_new / 𝐾𝐿) - 1
    # Update capital-labor ratio using weighted average formula
    𝐾𝐿 = (1 - step_KL) * 𝐾𝐿 + step_KL * 𝐾𝐿_new
    # Update iteration number
    iter_KL = iter_KL + 1
end;

## Results

# Equilibrium factor prices
𝑟 = r(𝐾𝐿);
𝑤 = w(𝐾𝐿);
𝑇 = τ * 𝑤 * sum(𝐙 .* 𝜋);
𝐾𝐿_demand = (α / (1 - α)) * (𝑤 / (𝑟 + δ));
𝐀_mean = mean(𝐀_sim[:, 𝐓]);
𝐀_median = quantile(𝐀_sim[:, 𝐓], 0.5);
𝐀_99 = quantile(𝐀_sim[:, 𝐓], 0.99);
𝐀_99_50 = 𝐀_99 / 𝐀_median;
𝐀_zero = count(i -> i == 𝑎̲, 𝐀_sim[:, 𝐓]) / length(𝐀_sim[:, 𝐓]);

# Table with results
key = [L"r", L"w", L"T", L"K/L", "Mean wealth", "Median wealth", "99th percentile", "99th-50th ratio", "Fraction with zero wealth"];
table = TableCol(L"\tau=0.15", key, [𝑟; 𝑤; 𝑇; 𝐾𝐿_demand; 𝐀_mean; 𝐀_median; 𝐀_99; 𝐀_99_50; 𝐀_zero]);
to_tex(table) |> print

# Plot consumption policy function
plot(
    𝐀,
    𝐂[:, 1],
    xrange = (𝑎̲, 𝑎̅),
    color = :blue,
    legend = :bottomright,
    label = "Lowest Income State",
    xlabel = "Assets",
    ylabel = "Consumption",
);
plot!(
    𝐀,
    𝐂[:, 5],
    xrange = (𝑎̲, 𝑎̅),
    color = :red,
    legend = :bottomright,
    label = "Highest Income State",
    xlabel = "Assets",
    ylabel = "Consumption",
);
savefig("consumption_policy_function_income_state.png");

# Plot savings policy function
𝐒_rate = Matrix(undef, 𝑁, 𝑛)
for 𝑧 = 1:𝑛
    for 𝑎 = 1:𝑁
        𝐒_rate[𝑎, 𝑧] = 𝐒[𝑎, 𝑧] - 𝐀[𝑎]
    end
end;
plot(
    𝐀,
    𝐒_rate[:, 1],
    xrange = (𝑎̲, 𝑎̅),
    color = :blue,
    legend = :topright,
    label = "Lowest Income State",
    xlabel = "Assets",
    ylabel = "Savings",
);
plot!(
    𝐀,
    𝐒_rate[:, 5],
    xrange = (𝑎̲, 𝑎̅),
    color = :red,
    legend = :topright,
    label = "Highest Income State",
    xlabel = "Assets",
    ylabel = "Savings",
);
savefig("savings_policy_function_income_state.png");

# Plot stationary wealth distribution
histogram(
    𝐀_sim[:, 𝐓],
    xrange = (minimum(𝐀_sim[:, 𝐓]), maximum(𝐀_sim[:, 𝐓])),
    legend = false,
    color = :grey,
    xlabel = "Assets",
    ylabel = "Frequency",
);
savefig("wealth_distribution.png");
