function kern_rbf(𝑥ᵢ, 𝑥ⱼ;
                  ls = NaN,
                  v  = NaN,
                  θ::Dict = Dict("lengthscale" => 1.,
                                 "variance"    => 1.))


    if !isnan(ls)
        θ["lengthscale"] = convert(Float64,ls)
    end

    if !isnan(v)
        θ["variance"]    = convert(Float64, v)
    end

    if length(𝑥ᵢ) != length(𝑥ⱼ)
        println("error: input dimensions incorrect")
        return NaN
    end

    𝑘 = θ["variance"] .* exp(-norm(𝑥ᵢ - 𝑥ⱼ).^2 ./ θ["lengthscale"]^2)

    return 𝑘
end

####################################################################################

function kern_linear(𝑥ᵢ, 𝑥ⱼ;
                     b = NaN,
                     v = NaN,
                     c = NaN,
                     θ::Dict{String,Float64} = Dict("constant_variance" => 0.,
                                                    "offset"      => 1.,
                                                    "variance"    => 1.))

    if !isnan(b)
        θ["constant_variance"] = convert(Float64,b)
    end
    if !isnan(c)
        θ["offset"]            = convert(Float64,c)
    end
    if !isnan(v)
        θ["variance"]          = convert(Float64, v)
    end

    𝑘 = θ["constant_variance"] 
            + θ["variance"] * (𝑥ᵢ - θ["offset"]) * (𝑥ⱼ - θ["offset"])

    return 𝑘
end

####################################################################################

function kern_periodic(𝑥ᵢ, 𝑥ⱼ;
                       p  = NaN,
                       ls = NaN,
                       v  = NaN,
                       θ::Dict{String,Float64} = Dict("period"      => 1., 
                                                      "lengthscale" => 0.1,
                                                      "variance"    => 1.))
    if !isnan(p)
        θ["period"] = convert(Float64,p)
    end
    if !isnan(ls)
        θ["lengthscale"] = convert(Float64,ls)
    end
    if !isnan(v)
        θ["variance"]    = convert(Float64, v)
    end

    𝑘 = θ["variance"] * exp(-(2*sin(π*norm(𝑥ᵢ - 𝑥ⱼ)/θ["period"])^2) 
                              / θ["lengthscale"]^2)

    return 𝑘

end

####################################################################################

function kern_exponential_𝛾(𝑥ᵢ, 𝑥ⱼ;
                            𝛾  = NaN,
                            ls = NaN,
                            v  = NaN,
                            θ::Dict{String,Float64} = Dict("exponent"    => 1., 
                                                           "lengthscale" => 0.1,
                                                           "variance"    => 1.))
    if !isnan(𝛾)
        θ["exponent"] = convert(Float64, 𝛾)
    end
    if !isnan(ls)
        θ["lengthscale"] = convert(Float64, ls)
    end
    if !isnan(v)
        θ["variance"]    = convert(Float64, v)
    end

    𝑘 = θ["variance"] .* exp(-(norm(𝑥ᵢ - 𝑥ⱼ)./θ["lengthscale"]).^θ["exponent"])

    return 𝑘
end
####################################################################################

function kern_exponential(𝑥ᵢ, 𝑥ⱼ;
                          ls = NaN,
                          v  = NaN,
                          θ::Dict{String,Float64} = Dict("lengthscale" => 0.1,
                                                           "variance"    => 1.))
    return kern_exponential_𝛾(𝑥ᵢ, 𝑥ⱼ, 𝛾 = 1., ls = ls, v = v, θ = θ)
end

####################################################################################

function sample_kernel(𝐱; 𝑘 = kern_rbf, θ = nothing)

    𝑛ₓ = length(𝐱)

    if θ == nothing
        𝐊ₛₛ = reshape([𝑘(𝑥ᵢ, 𝑥ⱼ) for 𝑥ᵢ ∈ 𝐱 for 𝑥ⱼ ∈ 𝐱], 𝑛ₓ, 𝑛ₓ)
    else
        𝐊ₛₛ = reshape([𝑘(𝑥ᵢ, 𝑥ⱼ, θ=θ) for 𝑥ᵢ ∈ 𝐱 for 𝑥ⱼ ∈ 𝐱], 𝑛ₓ, 𝑛ₓ)
    end


    U,S,V = svd(𝐊ₛₛ)
    𝐿 = U * diagm(sqrt.(S))

    return 𝐿*randn((𝑛ₓ,1))
end

####################################################################################

function sample_nd_kernel(𝐮; 𝑘 = kern_rbf, θ = nothing)

    𝑛ᵤ = length(𝐮)
    𝑛ₓ = length(𝐮[1][:])

    #iterator (2d only)
    u_itr = zip(𝐮[1][:],𝐮[2][:])

    if θ == nothing
        𝐊ₛₛ = reshape([𝑘(collect(𝑥ᵢ), collect(𝑥ⱼ))
                        for 𝑥ᵢ ∈ u_itr for 𝑥ⱼ ∈ u_itr], 𝑛ₓ, 𝑛ₓ)
    else
        𝐊ₛₛ = reshape([𝑘(collect(𝑥ᵢ), collect(𝑥ⱼ), θ=θ)
                        for 𝑥ᵢ ∈ u_itr for 𝑥ⱼ ∈ u_itr], 𝑛ₓ, 𝑛ₓ)
    end

    U,S,V = svd(𝐊ₛₛ)
    𝐿 = U * diagm(sqrt.(S))

    return 𝐿*randn((𝑛ₓ,1))
end

####################################################################################

function trained_gp(uₛ,yₛ;
                    σ² = 0., # default noise-free
                    𝑘 = kern_rbf,
                    θ = Dict("lengthscale"=>0.25, "variance"=>1.))

    𝑛ₛ   = size(uₛ,1)

    𝐊ₛₛ  = zeros(𝑛ₛ, 𝑛ₛ)
    for 𝑖 ∈ 1:𝑛ₛ
        for 𝑗 ∈ 1:𝑛ₛ
            𝐊ₛₛ[𝑖,𝑗] = 𝑘(uₛ[𝑖,:],uₛ[𝑗,:],θ=θ)
        end
    end

    return u -> _proto_gp_predict(u, uₛ, yₛ, 𝐊ₛₛ + σ²*eye(𝑛ₛ), 𝑘, θ)
end

function _proto_gp_predict(u, uₛ, yₛ, 𝐊ₛₛ, 𝑘, θ)

    𝑛ₛ  = size(uₛ,1)
    𝑛₊  = size(u,1)

    𝐊ₛ₊  = zeros(𝑛₊, 𝑛ₛ)
    for 𝑖 ∈ 1:𝑛₊
        for 𝑗 ∈ 1:𝑛ₛ
            𝐊ₛ₊[𝑖,𝑗] = 𝑘(u[𝑖,:],uₛ[𝑗,:],θ=θ)
        end
    end

    𝐊₊₊  = zeros(𝑛₊, 𝑛₊)
    for 𝑖 ∈ 1:𝑛₊
        for 𝑗 ∈ 1:𝑛₊
            𝐊₊₊[𝑖,𝑗] = 𝑘(u[𝑖,:],u[𝑗,:],θ=θ)
        end
    end

    𝛍 = (𝐊ₛ₊ / 𝐊ₛₛ) * yₛ
    𝚺 = 𝐊₊₊ - (𝐊ₛ₊ / 𝐊ₛₛ) * 𝐊ₛ₊'

    return 𝛍, 𝚺
end

####################################################################################

function sample_posterior(μ, Σ; n=1)

    𝑛ₓ = size(Σ,1)

    # Calculate lower triangular root
    U,S,V = svd(Σ)
    𝐿 = U * diagm(sqrt.(S))

    return [μ + 𝐿*randn(𝑛ₓ) for _ in 1:n]
end