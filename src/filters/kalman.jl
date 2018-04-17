struct State
    𝔼   :: Vector{Float64}
    cov :: Matrix{Float64}
end

type KalmanFilter <: _Filter

    # States
    𝐱ₜ   :: State
    𝐱₀₋ₜ :: Array{State}

    # Dynamic process
    𝐅  :: Matrix{Float64}
    𝐔ᶠ :: Matrix{Float64}

    # Observation
    𝐇 :: Matrix{Float64}
    𝐔ʰ :: Matrix{Float64}

    # Noise terms
    𝐐 :: Matrix{Float64}
    𝐑 :: Matrix{Float64}

    # Dimensions
    𝑛ˣ :: Int
    𝑛ʸ :: Int
    𝑡  :: Int

    function KalmanFilter(𝐱₀, 𝐏₀, 𝐅, 𝐐, 𝐔ᶠ, 𝐇, 𝐑, 𝐔ʰ)
        if !(typeof(𝐱₀) <: Array)
            𝐱₀ = Vector{Float64}([𝐱₀])
        end
        𝑛ˣ = length(𝐱₀)

        𝐏₀ = at_least2d(𝐏₀)
        𝐅  = at_least2d(𝐅)
        𝐔ᶠ = at_least2d(𝐔ᶠ)
        𝐐  = at_least2d(𝐐)
        𝐇  = at_least2d(𝐇)
        𝐔ʰ = at_least2d(𝐔ʰ)
        𝐑  = at_least2d(𝐑)

        𝑛ʸ = size(𝐇,1)

        𝐱 = State(𝐱₀, 𝐏₀)
        kf = new(𝐱, Array{State}([𝐱]),
                 𝐅, 𝐔ᶠ,
                 𝐇, 𝐔ʰ,
                 𝐐, 𝐑,
                 𝑛ˣ, 𝑛ʸ, 1)
        return kf
    end

    function KalmanFilter(𝐱₀, 𝐏₀)
        if !(typeof(𝐱₀) <: Array)
            𝐱₀ = Vector{Float64}([𝐱₀])
        end
        𝑛ˣ = length(𝐱₀)

        return KalmanFilter(𝐱₀, 𝐏₀, eye(𝑛ˣ), zeros(𝑛ˣ,𝑛ˣ), zeros(𝑛ˣ,𝑛ˣ), eye(𝑛ˣ), zeros(𝑛ˣ,𝑛ˣ), zeros(𝑛ˣ,𝑛ˣ))

    end

    function KalmanFilter(𝐱₀, 𝐏₀, 𝐅, 𝐐)
        if !(typeof(𝐱₀) <: Array)
            𝐱₀ = Vector{Float64}([𝐱₀])
        end
        𝑛ˣ = length(𝐱₀)

        return KalmanFilter(𝐱₀, 𝐏₀, 𝐅, 𝐐, zeros(𝑛ˣ,𝑛ˣ), eye(𝑛ˣ), zeros(𝑛ˣ,𝑛ˣ), zeros(𝑛ˣ,𝑛ˣ))

    end

    function KalmanFilter(𝐱₀, 𝐏₀, 𝐅, 𝐐, 𝐔ᶠ)
        if !(typeof(𝐱₀) <: Array)
            𝐱₀ = Vector{Float64}([𝐱₀])
        end
        𝑛ˣ = length(𝐱₀)

        return KalmanFilter(𝐱₀, 𝐏₀, 𝐅, 𝐐, 𝐔ᶠ, eye(𝑛ˣ), zeros(𝑛ˣ,𝑛ˣ), zeros(𝑛ˣ,𝑛ˣ))

    end

    function KalmanFilter(𝐱₀, 𝐏₀, 𝐅, 𝐐, 𝐇, 𝐑)
        if !(typeof(𝐱₀) <: Array)
            𝐱₀ = Vector{Float64}([𝐱₀])
        end
        𝑛ˣ = length(𝐱₀)

        return KalmanFilter(𝐱₀, 𝐏₀, 𝐅, 𝐐, zeros(𝑛ˣ,𝑛ˣ), 𝐇, 𝐑, zeros(𝑛ˣ,𝑛ˣ))

    end

end

function predict!(kf::KalmanFilter, 𝐮 = nothing, 𝐐 = nothing)
    𝑥ₜ₋₁ = kf.𝐱ₜ.𝔼
    𝑃ₜ₋₁ = kf.𝐱ₜ.cov

    if 𝐮 ≠ nothing
        𝑥ₜ = feval(kf.𝐅, 𝑥ₜ₋₁, kf.𝐔ᶠ * 𝐮)
    else
        𝑥ₜ = feval(kf.𝐅, 𝑥ₜ₋₁)
    end

    if 𝐐 ≠ nothing
        kf.𝐐 = 𝐐
    end
    
    𝑃ₜ = kf.𝐅 * 𝑃ₜ₋₁ * kf.𝐅' + kf.𝐐

    kf.𝐱ₜ = State(𝑥ₜ, 𝑃ₜ)
    push!(kf.𝐱₀₋ₜ, kf.𝐱ₜ)
    kf.𝑡 = length(kf.𝐱₀₋ₜ)
end

function update!(kf::KalmanFilter, 𝐲, 𝐮 = nothing, 𝐑 = nothing)
    𝑥ₜ = kf.𝐱ₜ.𝔼
    𝑃ₜ = kf.𝐱ₜ.cov

    if 𝐮 ≠ nothing
        𝑦ₜ = feval(kf.𝐇, 𝑥ₜ, kf.𝐔ʰ * 𝐮)
    else
        𝑦ₜ = feval(kf.𝐇, 𝑥ₜ)
    end

    if 𝐑 ≠ nothing
        kf.𝐑 = 𝐑
    end

    𝑃ˣʸ = 𝑃ₜ * kf.𝐇'
    𝑃ʸ = kf.𝐇 * 𝑃ˣʸ + kf.𝐑

    𝐊ₜ = 𝑃ˣʸ / 𝑃ʸ

    # 𝑥ₜ = 𝑥ₜ + 𝐊ₜ * (𝐲 - 𝑦ₜ)
    # 𝑃ₜ = 𝑃ₜ - 𝐊ₜ * 𝑃ʸ * 𝐊ₜ'
    𝑥ₜ += 𝐊ₜ * (𝐲 - 𝑦ₜ)
    𝑃ₜ -= 𝐊ₜ * 𝑃ʸ * 𝐊ₜ'

    kf.𝐱ₜ = State(𝑥ₜ, 𝑃ₜ)
    kf.𝐱₀₋ₜ[kf.𝑡] = kf.𝐱ₜ
end
