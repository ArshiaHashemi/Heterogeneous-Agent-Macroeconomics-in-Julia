################################################################################
#
#   PURPOSE: Rouwenhort Discretization of AR(1) Process for Labor Income
#   AUTHOR: Arshia Hashemi
#   EMAIL: arshiahashemi@uchicago.edu
#   DATE: Spring 2022
#
################################################################################

# Rouwenhorst
struct rouwenhorst
    # Define output
    𝐲::Vector{Float64} # Grid for income
    𝐏::Matrix{Float64} # Transition probability matrix
    𝛑::Vector{Float64} # Stationary probability mass function
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
            𝐏 = Matrix{Float64}(undef, 𝑛, 𝑛)
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
