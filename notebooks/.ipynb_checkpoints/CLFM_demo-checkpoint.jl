
#===================================
Kalman filter and smoother functions
====================================#

function _proto_predict(m, P, F, Q)
    """ Prototype for Kalman predict """
    return F*m, Hermitian(F*P*F'+Q)
end

function _proto_update(m, P, y, H, R)
    """ Prototype for Kalman update """
    K = P*H'/(H*P*H'+R)
    return m+K*(y-H*m), Hermitian(P-K*(H*P*H'+R)*K')
end

function _proto_smooth(m⁻,m⁺,P⁻,P⁺, F, Q)
    """ Prototype for RTS smooth """
    G = (P⁻*F'/(F*P⁻*F'+Q))
    return m⁻+ G*(m⁺-F*m⁻), Hermitian(P⁻+G*(P⁺-F*P⁻*F'+Q)*G')
end

function filter(F, Q, H, R)
    """ Kalman filter functions
    
    Input:
        - F   Time-invariant transition process
        - Q   Transition noise covariance
        - H   Time-invariant observation process
        - R   Observation noise covariance
    
    Output:
        - predict   Generic function of 2 inputs that takes in (m, P), the
                     mean and covariance of the previous state and returns a
                      prediction propagated by F
        - update    Generic function of 3 inputs that takes in (m, P, y), 
                     the (predicted) mean and covariance of the current state
                      and a current observation, and returns an updated estimate
    """
    return (m, P)    -> _proto_predict(m, P, F, Q),
           (m, P, y) -> _proto_update(m, P, y, H, R)
end

function smoother(F, Q)
    """ Rauch-Tung-Streibel smoother function 
    
    Input:
        - F   Time-invariant transition process
        - Q   Transition noise covariance
    
    Output:
        - smooth   Generic function of 4 inputs that takes in (m⁻,m⁺,P⁻,P⁺),
                    the means and covariances of the current and proceeding
                     (smoothed) state estimates and returns a smoothed estimate
    """
    return (m⁻,m⁺,P⁻,P⁺) -> _proto_smooth(m⁻,m⁺,P⁻,P⁺, F, Q)
end

#=

=#

function default_params(;as_dict::Bool=:true, key=nothing)
    """ Default parameters for the LFM example
    
    Input:
        - as_dict (optional, keyword)   boolean flag for output as dictionary [default: true]
        - key (optional, keyword)       key for parameters, overrides as_dict [default: nothing]
    
    Output:
        - params                        dictionary or tuple containing all (unless specified by key) params
    """
    
    a₁, a₂, a₃ = -1./2., -1./4., -1./6. # output rates of tanks
    𝐟₀ = [2, 5, 2.5]                    # initial concentration
    𝑉  = 4.                             # peak concentration of input 
    l, σₖ² = 0.5, 𝑉                     # lengthscale / variance of GP kernel
    λ = √3 / l                          # SSM kernel param
    𝑹 = 0.001^2                         # Observation noise
    tₛₜₐᵣₜ = [0.5, 2.5, 4.5]            # Injection start points
    tₛₜₒₚ  = [1.5, 3.5, 5.5]            # Injection end points
    σᵤ = 100.                           # Injection gradient
    
    if as_dict || key ≠ nothing
        𝛉 = Dict{String, Any}(
            "Output Rates"          => (a₁, a₂, a₃),
            "Initial Concentration" => 𝐟₀,
            "Input Parameters"      => (𝑉, tₛₜₐᵣₜ, tₛₜₒₚ, σᵤ),
            "Kernel Parameters"     => (λ, l, σₖ²),
            "Observation Noise"     => 𝑹
        )
        if key ≠ nothing
            return 𝛉[key]
        end
    else
        𝛉 = (a₁, a₂, a₃, 𝐟₀, λ, l, σₖ², 𝑹, 𝑉, tₛₜₐᵣₜ, tₛₜₒₚ, σᵤ)
    end
    
    return 𝛉
end

function _proto_input(θᵤ = default_params(key="Input Parameters"))
    """ Prototype of input function 
    
        Input:
            - params (optional)   tuple of parameters
        
        Ouput:
            - u(t)                generic function of t describing input
            - ∫u                  generic function of t describing the integral of input
                                    (use for discrete-time system)
    """
    𝑉, tₛₜₐᵣₜ, tₛₜₒₚ, σᵤ = θᵤ
    
    return t -> 𝑉*sum([(1 + exp.(σᵤ*(a-t))).^(-1).*(1 + exp.(σᵤ*(t-b))).^(-1) for (a,b) ∈ zip(tₛₜₐᵣₜ, tₛₜₒₚ)]),
           t -> 𝑉*sum([(log.(exp.(σᵤ*(a-b))+exp.(σᵤ*(a-t)))-log.(exp.(σᵤ*(a-t))+1))/(σᵤ*(exp(σᵤ*(a-b))-1)) for (a,b) ∈ zip(tₛₜₐᵣₜ, tₛₜₒₚ)])
end

#=

=#
function generate_observations(𝐭ₛ; 𝛉=default_params(as_dict=:true))
    """
    """
    
    
    nₛ = length(𝐭ₛ)
    𝐲ₜ = Array{Float64}(nₛ)
    
    a₁, a₂, a₃ = isa(𝛉, Dict) ? 𝛉["Output Rates"] : 𝛉[1:3]
    𝐟₀ = isa(𝛉, Dict) ? 𝛉["Initial Concentration"] : 𝛉[4]
    
    _, ∫𝑢 = _proto_input(isa(𝛉, Dict) ? 𝛉["Input Parameters"] : 𝛉[9:12])
    
    
    𝐀 = [ a₁  0.  0. ;
         -a₁  a₂  0. ;
          0. -a₂  a₃ ]
    𝐋 = [1 ; 0 ; 0]

    𝐇  = [0 0 -a₃]
    𝑹  = isa(𝛉, Dict) ? 𝛉["Observation Noise"] : 𝛉[8]
    
    𝑓ₜ = 𝐟₀
    𝐲ₜ[1] = (𝐇*𝐟₀)[1]

    for k ∈ 2:nₛ
        𝑓ₜ = expm((𝐭ₛ[k]-𝐭ₛ[k-1])*𝐀) * 𝑓ₜ + 𝐋*(∫𝑢(𝐭ₛ[k]) - ∫𝑢(𝐭ₛ[k-1]))
        𝐲ₜ[k] = (𝐇*𝑓ₜ + √𝑹 * randn())[1]
    end
    
    return 𝐲ₜ
end

function generate_solution(𝐭; 𝛉 = default_params(as_dict=:true))
    """
    
    """
    
    nₜ = length(𝐭)
    a₁, a₂, a₃ = isa(𝛉, Dict) ? 𝛉["Output Rates"] : 𝛉[1:3]
    𝐟₀ = isa(𝛉, Dict) ? 𝛉["Initial Concentration"] : 𝛉[4]
    
    _, ∫𝑢 = _proto_input(isa(𝛉, Dict) ? 𝛉["Input Parameters"] : 𝛉[9:12])
    
    𝐀 = [ a₁  0.  0. ;
         -a₁  a₂  0. ;
          0. -a₂  a₃ ]
    𝐋 = [1 ; 0 ; 0]
    
    𝒇ₜ = Array{typeof(𝐟₀)}(nₜ)
    𝒇ₜ[1] = 𝐟₀
    for k ∈ 2:nₜ
        𝒇ₜ[k] = expm((𝐭[k]-𝐭[k-1])*𝐀) * 𝒇ₜ[k-1] + 𝐋*(∫𝑢(𝐭[k]) - ∫𝑢(𝐭[k-1]))
    end
    
    return 𝒇ₜ
end


#=

=#
function system_model(; 𝛉 = default_params(as_dict=:true))
    """
    """
    a₁, a₂, a₃ = isa(𝛉, Dict) ? 𝛉["Output Rates"] : 𝛉[1:3]
    λ, l, σₖ² = isa(𝛉, Dict) ? 𝛉["Kernel Parameters"] : 𝛉[5:7]
    
    𝐀 = [   0.   1. 0.  0.  0. ;
           -λ^2 -2λ 0.  0.  0. ;
            1.   0. a₁  0.   0.;
            0.   0. -a₁  a₂  0.;
            0.   0. 0. -a₂  a₃ ]
    
    P∞ = diagm([σₖ², λ^2*σₖ²])
    
    𝐇 = [0 0 0 0 -a₃]
    𝑹  = isa(𝛉, Dict) ? 𝛉["Observation Noise"] : 𝛉[8]
    
    return 𝐀, P∞, 𝐇, 𝑹
end


#=

=#
include("GP.jl")
using DataFrames
function demo(; 𝛉=default_params(as_dict=:true), as_dataframe::Bool=:false)
    """ Demo function
    
    """
    
    ## Discretise system
    nₜ = 1000
    𝐭 = linspace(0., 7., nₜ)
    
    # Observe system
    𝐭ₛ = 𝐭[1:10:end]
    𝐲ₜ = generate_observations(𝐭ₛ, 𝛉=𝛉)
    
    # Setup filtering models
    𝐟, 𝐏, 𝐟⁻, 𝐏⁻, 𝐆 = infer(𝐭, 𝐲ₜ, 𝐭ₛ, 𝛉=𝛉, rts=:true, gain=:true)
    
    p = plot_estimate(𝐟⁻, 𝐏⁻, 𝐭, show_truth=:true, title="Filtered estimate", 𝛉=𝛉)
    draw(SVG("CLFM_demo_filtered.svg", 24cm, 30cm), p)
    
    p = plot_estimate(𝐟, 𝐏, 𝐭, show_truth=:true, title="Full posterior estimate", 𝛉=𝛉)
    draw(SVG("CLFM_demo_smoothed.svg", 24cm, 30cm), p)
    #=
    𝚺 = zeros(nₜ,nₜ)
    for i ∈ 1:nₜ
        for j ∈ 1:nₜ
            𝚺[i,j] = i == j ? 𝐏[i][1,1] : prod([G[1,1] for G ∈ 𝐆[i:j-1]])*𝐏[i][1,1] 
        end
    end
    
    p = plot_samples(𝐟, 𝚺)
    draw(SVG("CLFM_demo_input_samples.svg", 24cm, 10cm), p)
    =#
    if as_dataframe
        return DataFrame(Dict(
                lbl=>val 
                for (lbl,val) = zip(
                        [:t, :mean, :cov, :mean_filter, :cov_filter, :gain],
                        (𝐭, 𝐟, 𝐏, 𝐟⁻, 𝐏⁻, 𝐆))))
    else
        return 𝐭, 𝐟, 𝐏, 𝐟⁻, 𝐏⁻, 𝐆
    end
end

#=

=#
using Gadfly
function plot_estimate(f=nothing, P=nothing, t=0.; show_truth = :true, title::String="", 𝛉=default_params(as_dict=:true))
    """
    """
    p = Array{Plot}(4)

    if show_truth
        𝑢, _ = _proto_input(isa(𝛉, Dict) ?𝛉["Input Parameters"] : 𝛉[9:12])
        𝒇ₜ   = generate_solution(t; 𝛉=𝛉)
    end
    
    p[1] = plot(
        show_truth ? layer(
            x=t,
            y=𝑢(t),
            color=[colorant"black" for _ ∈ t],
            Geom.line,
            style(line_style=:dash)
        ) : nothing,
        f ≠ nothing ? layer(
            x=t,
            y=[f_[1] for f_ ∈ f],
            ymin=[f_[1] - 1.96*√abs(σ[1]) for (f_,σ) ∈ zip(f, P)],
            ymax=[f_[1] + 1.96*√abs(σ[1]) for (f_,σ) ∈ zip(f, P)],
            color=[colorant"blue" for _ ∈ t],
            Geom.line, Geom.ribbon
        ) : nothing,
        Coord.cartesian(xmin=0., xmax=t[end]),
        Guide.xlabel("𝑡"),
        Guide.ylabel("𝑢(𝑡)"),
        Guide.title("Concentration of injected brine"),
        Guide.manual_color_key("", [show_truth ? "true input" : nothing, f ≠ nothing ? "filtered estimate" : nothing], [show_truth ? "black" : nothing, f ≠ nothing ? "blue" : nothing])
    )

    for i ∈ 1:3
        p[i+1] = plot(
            show_truth ? layer(
                x=t, y=[f[i] for f ∈ 𝒇ₜ],
                color=[colorant"black" for _ ∈ t],
                Geom.line,
                style(line_style=:dash)
            ) : nothing,
            f ≠ nothing ? layer(
                x=t,
                y=[f_[i+2] for f_ ∈ f],
                ymin=[f_[i+2] - 1.96*√abs(σ[i+2]) for (f_,σ) ∈ zip(f, P)],
                ymax=[f_[i+2] + 1.96*√abs(σ[i+2]) for (f_,σ) ∈ zip(f, P)],
                color=[colorant"blue" for _ ∈ t],
                Geom.line, Geom.ribbon
            ) : nothing,
           # x=:x, y=:y, color=:label, Geom.line,
            Coord.cartesian(xmin=0., xmax=maximum(t)),
            Guide.xlabel("𝑡"),
            Guide.ylabel(i == 1 ? "𝑓₁(𝑡)" : i == 2 ? "𝑓₂(𝑡)" : "𝑓₃(t)"),
            Guide.title(@sprintf("Concentration in tank %d", i)),
            Guide.manual_color_key("", ["true concentration", "filtered estimate"], ["black","blue"])
        )
    end
    
    return vstack(p)
end

function plot_samples()
    """
    """
end

function infer(𝐭, 𝐲ₜ, 𝐭ₛ; 𝛉=default_params(as_dict=:true), rts::Bool=:true, gain::Bool=:false)
    """
    """

    # Latent force kernel
    𝐀, P∞, 𝐇, 𝑹 = system_model(𝛉=𝛉)
    
    nₜ = length(𝐭)
    Δₜ = 𝐭[2] - 𝐭[1]
    
    𝐟⁻ = inv(expm(Δₜ*𝐀[3:end,3:end])) * (isa(𝛉, Dict) ? 𝛉["Initial Concentration"] : 𝛉[4])
    𝐟ᵃ₀ = [0.; 0.; 𝐟⁻] # initial conditions
    
    𝐅 = expm(Δₜ*𝐀[1:2,1:2])
    𝑸 = [P∞ - 𝐅*P∞*𝐅' zeros(2,3) ; zeros(3, 5) ] + eps()*eye(5)
    
    𝐅 = expm(Δₜ*𝐀)
    m₀, P₀ = 𝐟ᵃ₀, [P∞ zeros(2,3) ; zeros(3, 5)]
    
    # Setup
    𝐟, 𝐏 = Array{typeof(m₀)}(nₜ), Array{typeof(P₀)}(nₜ)
    
    # Kalman filtering
    predict, update = filter(𝐅, 𝑸, 𝐇, 𝑹)
    mₜ, Pₜ = m₀, P₀
    for k ∈ 1:nₜ
        mₜ, Pₜ = predict(mₜ, Pₜ)

        if 𝐭[k] ∈ 𝐭ₛ
            j = find(𝐭[k] .== 𝐭ₛ)[1]
            mₜ, Pₜ = update(mₜ, Pₜ, 𝐲ₜ[j])
        end
        𝐟[k] = mₜ
        𝐏[k] = Pₜ
    end

    if rts
        # Copy filtered approximations
        𝐟⁻, 𝐏⁻ = copy(𝐟), copy(𝐏)
        
        # RTS smoothing
        smooth = smoother(𝐅, 𝑸)
        
        gain ? 𝐆 = Array{typeof(𝑸)}(nₜ) : nothing

        for k = nₜ-1:-1:1
            gain ? 𝐆[k] = (𝐏[k]*𝐅'/(𝐅*𝐏[k]*𝐅'+𝑸)) : nothing
            𝐟[k], 𝐏[k] = smooth(𝐟[k],𝐟[k+1],𝐏[k],𝐏[k+1]) 
        end
    end
    
    return rts ? (gain ? (𝐟, 𝐏, 𝐟⁻, 𝐏⁻, 𝐆) : (𝐟, 𝐏, 𝐟⁻, 𝐏⁻)) : (𝐟, 𝐏)
end

#=

=#
using Distributions

function likelihood(f, P; 𝐆 = nothing, 𝛉=default_params(as_dict=:true), noiseless::Bool=:false)
    """
    """
    _, _, 𝐇, 𝑹 = system_model(𝛉 = 𝛉)
    
    noiseless ? 𝑹 = 0. : nothing
    
    μ = [(𝐇*𝑓ₜ)[1] for 𝑓ₜ ∈ f]
    
    if 𝐆 ≠ nothing
        nₜ = length(f)
        
        𝚺 = zeros(nₜ,nₜ)
        for i ∈ 1:nₜ
            for j ∈ 1:i
                𝚺[i,j] = (
                    i == j ? 𝐇*P[i]*𝐇'+ 𝑹 : 𝐇*prod([G for G ∈ 𝐆[j:i-1]])*P[i]*𝐇'
                )[1]
            end
        end
        for i ∈ 1:nₜ
            for j ∈ i+1:nₜ
                𝚺[i,j] = 𝚺[j,i]
            end
        end
    else
       𝚺 = diagm([(𝐇*Pₜ*𝐇' + 𝑹)[1] for Pₜ ∈ P])
    end
    
    𝚺 = Hermitian(𝚺)
    try
        assert(isposdef(𝚺))
    catch e
        if isa(e, AssertionError)
            println("assertion error: likelihood covariance non-positive-definite")
            println(𝛉["Kernel Parameters"][1])
            println(𝛉["Output Rates"])
            println(𝚺)
        end
    end
    
    return MvNormal(μ, 𝚺)
end