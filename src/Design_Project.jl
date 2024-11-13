module Design_Project

using DifferentialEquations
using Plots

export sirs_model!
export plot_model_vs_data
export compute_error
export compute_error_severe

# Function defining the SIRS model differential equations with intervention
function sirs_model!(dpop, pop, params, t)
    # Parameters
    c, β, γ, γ_s, ps, σ, ε, ϕ, intervention_day = params

    # Population numbers
    S, I, Is, R = pop

    # Total population
    N = S + I + Is + R

    # Apply the intervention after day 30
    if t > intervention_day
        λ = c * (1 - ε * ϕ) * β * I / N
    else
        λ = c * β * I / N
    end

    # Susceptible population dynamics
    dpop[1] = -λ * S + σ * R          # dS/dt

    # Infected population dynamics
    dpop[2] = λ * S - γ * I           # dI/dt

    # Severely infected population dynamics
    dpop[3] = γ * I * ps - γ_s * Is   # dIs/dt

    # Recovered population dynamics
    dpop[4] = γ * I * (1 - ps) + γ_s * Is - σ * R # dR/dt
end

# Function to compute the error for a given β, now including intervention parameters
function compute_error(β, params_fixed, initial_conditions, tspan, days, infected_people)
    # Unpack fixed parameters
    c, γ, γ_s, ps, σ, ε, ϕ, intervention_day = params_fixed
    # Complete parameter set with current β
    params = (c, β, γ, γ_s, ps, σ, ε, ϕ, intervention_day)

    # Solve the differential equations using an ODE solver
    prob = ODEProblem(sirs_model!, initial_conditions, tspan, params)
    sol = solve(prob)

    # Ensure 'days' is treated as an array
    days_array = collect(days)

    # Extract model predictions at the given days
    model_infected = sol(days_array)
    model_total_infected = model_infected[2, :]

    # Compute the sum of squared errors between model predictions and data
    summed = sum((model_total_infected - infected_people).^2)
    error = sqrt(summed/length(days))
    return error
end

function compute_error_severe(β, params_fixed, initial_conditions, tspan, days, infected_people)
    # Unpack fixed parameters
    c, γ, γ_s, ps, σ, ε, ϕ, intervention_day = params_fixed
    # Complete parameter set with current β
    params = (c, β, γ, γ_s, ps, σ, ε, ϕ, intervention_day)

    # Solve the differential equations using an ODE solver
    prob = ODEProblem(sirs_model!, initial_conditions, tspan, params)
    sol = solve(prob)

    # Ensure 'days' is treated as an array
    days_array = collect(days)

    # Extract model predictions at the given days
    model_infected = sol(days_array)
    model_total_infected = model_infected[3, :]

    # Compute the sum of squared errors between model predictions and data
    summed = sum((model_total_infected - infected_people).^2)
    error = sqrt(summed/length(days))
    return error
end

end 
