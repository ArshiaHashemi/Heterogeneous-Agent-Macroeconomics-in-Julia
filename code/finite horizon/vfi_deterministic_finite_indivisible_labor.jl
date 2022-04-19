################################################################################
#
#   PURPOSE: Finite Horizon Deterministic VFI with Indivisible Labor Supply
#   AUTHOR: Arshia Hashemi
#   EMAIL: arshiahashemi@uchicago.edu
#   DATE: Spring 2022
#
################################################################################

## Preamble

# Working directory
cd(
    "C:\\Users\\arshi\\Dropbox\\ArshiaHashemiRA\\ECMA 33620\\assignments\\assignment1\\output\\",
);

# Packages
using Plots

## Parameters

# Time horizon
𝑇 = 50;

# Coefficient of relative risk aversion
γ = 1.5;

# Discount rate
β = 0.98;

# Labor disutility
𝜓 = 0.5;

# Asset lower bound
𝑎̲ = 0.0;

# Asset upper bound
𝑎̄ = 15.0;

# Dimension of asset grid
𝑁 = 1000;

# Asset grid
𝐴 = collect(range(𝑎̲, stop = 𝑎̄, length = 𝑁));

## Primitive Functions

# Period utility function
function utility(𝑐, ℏ)
    if γ == 1
        𝑢 = log(𝑐) - (𝜓 * ℏ)
    elseif γ != 1
        𝑢 = (𝑐^(1 - γ) - 1) / (1 - γ) - (𝜓 * ℏ)
    end
    return 𝑢
end;

# Deterministic wage process
function wage(𝑡)
    if 𝑡 <= (𝑇 / 2)
        𝑤 = 𝑡 / 10
    elseif 𝑡 > (𝑇 / 2)
        𝑤 = (𝑇 + 1 - 𝑡) / 10
    end
    return 𝑤
end;

## Optimal Policy Functions

# Optimal polciy functions in terms of government policy variables
function policy(𝑟, τ, 𝑏)
    ## Initalize Matrices for Optimal Policy Functions
    # Next period's assets
    𝐀 = Matrix(undef, 𝑁, 𝑇)
    # Consumption
    𝐂 = Matrix(undef, 𝑁, 𝑇)
    # Labor supply
    𝐇 = Matrix(undef, 𝑁, 𝑇)
    # Value function
    𝐕 = Matrix(undef, 𝑁, 𝑇)
    ## Terminal Period
    # Loop over current period's state variable 𝑎
    for 𝑎 = 1:𝑁
        # Initalize matrices
        𝐶 = Matrix(undef, 𝑁, 2)
        𝑉 = Matrix(undef, 𝑁, 2)
        # Optimal next period's assets is zero
        𝐀[𝑎, 𝑇] = 0.0
        # Working
        𝐶[𝑎, 1] = (1 + 𝑟) * 𝐴[𝑎] + (1 - τ) * wage(𝑇) - 𝐀[𝑎, 𝑇]
        𝑉[𝑎, 1] = utility(max(𝐶[𝑎, 1], 1.0E-10), 1.0)
        # Not working
        𝐶[𝑎, 2] = (1 + 𝑟) * 𝐴[𝑎] + 𝑏 - 𝐀[𝑎, 𝑇]
        𝑉[𝑎, 2] = utility(max(𝐶[𝑎, 2], 1.0E-10), 0.0)
        # Optimal consumption and labor supply
        if 𝑉[𝑎, 1] >= 𝑉[𝑎, 2]
            𝐇[𝑎, 𝑇] = 1.0
            𝐂[𝑎, 𝑇] = 𝐶[𝑎, 1]
            𝐕[𝑎, 𝑇] = 𝑉[𝑎, 1]
        elseif 𝑉[𝑎, 1] < 𝑉[𝑎, 2]
            𝐇[𝑎, 𝑇] = 0.0
            𝐂[𝑎, 𝑇] = 𝐶[𝑎, 2]
            𝐕[𝑎, 𝑇] = 𝑉[𝑎, 2]
        end
    end
    ## Backward Induction Value Function Iteration
    # Loop over time periods starting in period 𝑡=𝑇-1
    for 𝑡 = (𝑇-1):-1:1
        # Loop over current period's state variable 𝑎
        for 𝑎 = 1:𝑁
            # Initalize matrices
            𝐶 = Matrix(undef, 𝑁, 2)
            𝑉 = Matrix(undef, 𝑁, 2)
            # Loop over next period's state variable 𝑎′
            for 𝑎′ = 1:𝑁
                # Working
                𝐶[𝑎′, 1] = (1 + 𝑟) * 𝐴[𝑎] + (1 - τ) * wage(𝑡) - 𝐴[𝑎′]
                𝑉[𝑎′, 1] = utility(max(𝐶[𝑎′, 1], 1.0E-10), 1.0) + β * 𝐕[𝑎′, 𝑡+1]
                # Not working
                𝐶[𝑎′, 2] = (1 + 𝑟) * 𝐴[𝑎] + 𝑏 - 𝐴[𝑎′]
                𝑉[𝑎′, 2] = utility(max(𝐶[𝑎′, 2], 1.0E-10), 0.0) + β * 𝐕[𝑎′, 𝑡+1]
            end
            # Index for optimal asset choice and labor supply
            𝐼 = argmax(𝑉)
            # Optimal asset choice index
            𝑎′ = 𝐼[1]
            # Optimal asset choice value
            𝐀[𝑎, 𝑡] = 𝐴[𝑎′]
            # Optimal consumption and labor supply
            if 𝐼[2] == 1
                𝐇[𝑎, 𝑡] = 1.0
                𝐂[𝑎, 𝑡] = (1 + 𝑟) * 𝐴[𝑎] + (1 - τ) * wage(𝑡) - 𝐀[𝑎, 𝑡]
            elseif 𝐼[2] == 2
                𝐇[𝑎, 𝑡] = 0.0
                𝐂[𝑎, 𝑡] = (1 + 𝑟) * 𝐴[𝑎] + 𝑏 - 𝐀[𝑎, 𝑡]
            end
            # Value function
            𝐕[𝑎, 𝑡] = utility(𝐂[𝑎, 𝑡], 𝐇[𝑎, 𝑡]) + β * 𝐕[𝑎′, 𝑡+1]
        end
    end
    # Return output
    return (𝐀, 𝐂, 𝐇, 𝐕)
end;

## Evaluate Optimal Policy Functions Under Different Policy Regimes

# Baseline: (𝑟=0.02,τ=0.0,𝑏=0.5)
baseline = policy(0.02, 0.0, 0.5);
𝐀1 = baseline[1];
𝐂1 = baseline[2];
𝐇1 = baseline[3];
𝐕1 = baseline[4];

# Counterfactual 1: Decrease in interest rate (𝑟=0.01,τ=0.0,𝑏=0.5)
counterfactual1 = policy(0.01, 0.0, 0.5);
𝐀2 = counterfactual1[1];
𝐂2 = counterfactual1[2];
𝐇2 = counterfactual1[3];
𝐕2 = counterfactual1[4];

# Counterfactual 2: Positive labor income tax (𝑟=0.02,τ=0.4,𝑏=0.5)
counterfactual2 = policy(0.02, 0.4, 0.5);
𝐀3 = counterfactual2[1];
𝐂3 = counterfactual2[2];
𝐇3 = counterfactual2[3];
𝐕3 = counterfactual2[4];

# Counterfactual 3: Decrease in unemployment benefit (𝑟=0.02,τ=0.0,𝑏=0.1)
counterfactual3 = policy(0.02, 0.0, 0.1);
𝐀4 = counterfactual3[1];
𝐂4 = counterfactual3[2];
𝐇4 = counterfactual3[3];
𝐕4 = counterfactual3[4];

## Plots

# Initalize life-cycle matrices
𝔸1 = Vector(undef, 𝑇 + 1);
𝔸2 = Vector(undef, 𝑇 + 1);
𝔸3 = Vector(undef, 𝑇 + 1);
𝔸4 = Vector(undef, 𝑇 + 1);
ℂ1 = Vector(undef, 𝑇);
ℂ2 = Vector(undef, 𝑇);
ℂ3 = Vector(undef, 𝑇);
ℂ4 = Vector(undef, 𝑇);
ℍ1 = Vector(undef, 𝑇);
ℍ2 = Vector(undef, 𝑇);
ℍ3 = Vector(undef, 𝑇);
ℍ4 = Vector(undef, 𝑇);

# Initial asset value
𝔸1[1] = 0.0;
𝔸2[1] = 0.0;
𝔸3[1] = 0.0;
𝔸4[1] = 0.0;

# Simulate policy functions
for 𝑡 = 1:𝑇
    # Asset index
    𝑎1 = searchsortedfirst(𝐴, 𝔸1[𝑡])
    𝑎2 = searchsortedfirst(𝐴, 𝔸2[𝑡])
    𝑎3 = searchsortedfirst(𝐴, 𝔸3[𝑡])
    𝑎4 = searchsortedfirst(𝐴, 𝔸4[𝑡])
    # Consumption
    ℂ1[𝑡] = 𝐂1[𝑎1, 𝑡]
    ℂ2[𝑡] = 𝐂2[𝑎2, 𝑡]
    ℂ3[𝑡] = 𝐂3[𝑎3, 𝑡]
    ℂ4[𝑡] = 𝐂4[𝑎4, 𝑡]
    # Labor supply
    ℍ1[𝑡] = 𝐇1[𝑎1, 𝑡]
    ℍ2[𝑡] = 𝐇2[𝑎2, 𝑡]
    ℍ3[𝑡] = 𝐇3[𝑎3, 𝑡]
    ℍ4[𝑡] = 𝐇4[𝑎4, 𝑡]
    # Next period's asset choice
    𝔸1[𝑡+1] = 𝐀1[𝑎1, 𝑡]
    𝔸2[𝑡+1] = 𝐀2[𝑎2, 𝑡]
    𝔸3[𝑡+1] = 𝐀3[𝑎3, 𝑡]
    𝔸4[𝑡+1] = 𝐀4[𝑎4, 𝑡]
end;

# Baseline
plot(
    ℂ1,
    ylabel = "Consumption / Wage",
    xlabel = "Age",
    color = :red,
    linestyle = :solid,
    label = "Consumption"
);
plot!(
    wage, 0, 50,
    ylabel = "Consumption / Wage",
    xlabel = "Age",
    color = :grey,
    linestyle = :solid,
    label = "Wage"
);
savefig("baseline_consumption");
plot(
    ℍ1,
    ylabel = "Labor Supply",
    xlabel = "Age",
    legend = false,
    color = :green,
    linestyle = :solid,
);
savefig("baseline_laborsupply");
plot(
    𝔸1,
    ylabel = "Assets",
    xlabel = "Age",
    legend = false,
    color = :blue,
    linestyle = :solid,
)
savefig("baseline_assets");

# Counterfactual 1
plot(
    ℂ1,
    ylabel = "Consumption",
    xlabel = "Age",
    color = :red,
    linestyle = :solid,
    legend = :bottomright,
    label = "Baseline",
);
plot!(
    ℂ2,
    ylabel = "Consumption",
    xlabel = "Age",
    color = :red,
    linestyle = :dash,
    legend = :bottomright,
    label = "Counterfactual",
);
savefig("counterfactual1_consumption");
plot(
    ℍ1,
    ylabel = "Labor Supply",
    xlabel = "Age",
    color = :green,
    linestyle = :solid,
    legend = false,
);
plot!(
    ℍ2,
    ylabel = "Labor Supply",
    xlabel = "Age",
    color = :green,
    linestyle = :dash,
    legend = false,
);
savefig("counterfactual1_laborsupply");
plot(
    𝔸1,
    ylabel = "Assets",
    xlabel = "Age",
    color = :blue,
    linestyle = :solid,
    legend = false,
);
plot!(
    𝔸2,
    ylabel = "Assets",
    xlabel = "Age",
    color = :blue,
    linestyle = :dash,
    legend = false,
);
savefig("counterfactual1_assets");

# Counterfactual 2
plot(
    ℂ1,
    ylabel = "Consumption",
    xlabel = "Age",
    color = :red,
    linestyle = :solid,
    legend = :bottomright,
    label = "Baseline",
);
plot!(
    ℂ3,
    ylabel = "Consumption",
    xlabel = "Age",
    color = :red,
    linestyle = :dash,
    legend = :bottomright,
    label = "Counterfactual",
);
savefig("counterfactual2_consumption");
plot(
    ℍ1,
    ylabel = "Labor Supply",
    xlabel = "Age",
    color = :green,
    linestyle = :solid,
    legend = false,
);
plot!(
    ℍ3,
    ylabel = "Labor Supply",
    xlabel = "Age",
    color = :green,
    linestyle = :dash,
    legend = false,
);
savefig("counterfactual2_laborsupply");
plot(
    𝔸1,
    ylabel = "Assets",
    xlabel = "Age",
    color = :blue,
    linestyle = :solid,
    legend = false,
);
plot!(
    𝔸3,
    ylabel = "Assets",
    xlabel = "Age",
    color = :blue,
    linestyle = :dash,
    legend = false,
);
savefig("counterfactual2_assets");

# Counterfactual 3
plot(
    ℂ1,
    ylabel = "Consumption",
    xlabel = "Age",
    color = :red,
    linestyle = :solid,
    legend = :bottomright,
    label = "Baseline",
);
plot!(
    ℂ4,
    ylabel = "Consumption",
    xlabel = "Age",
    color = :red,
    linestyle = :dash,
    legend = :bottomright,
    label = "Counterfactual",
);
savefig("counterfactual3_consumption");
plot(
    ℍ1,
    ylabel = "Labor Supply",
    xlabel = "Age",
    color = :green,
    linestyle = :solid,
    legend = false,
);
plot!(
    ℍ4,
    ylabel = "Labor Supply",
    xlabel = "Age",
    color = :green,
    linestyle = :dash,
    legend = false,
);
savefig("counterfactual3_laborsupply");
plot(
    𝔸1,
    ylabel = "Assets",
    xlabel = "Age",
    color = :blue,
    linestyle = :solid,
    legend = false,
);
plot!(
    𝔸4,
    ylabel = "Assets",
    xlabel = "Age",
    color = :blue,
    linestyle = :dash,
    legend = false,
);
savefig("counterfactual3_assets");
