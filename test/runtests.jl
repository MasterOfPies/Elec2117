using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Test
using DifferentialEquations
using Plots
using Statistics
using Design_Project

@testset "Design_Project.jl" begin
    # Fixed Parameters for the SIRS model (excluding β)
    c = 8                             # Contact rate
    γ = 1/7                           # Recovery rate for non-severe cases
    γ_s = 1/14                        # Recovery rate for severe cases
    ps = 0.2                          # Proportion of severe cases
    σ = 1/30                          # Resusceptibility rate
    ε = 0.3                           # Intervention efficacy
    ϕ = 0.8                           # Proportion that uses intervention
    intervention_day = 30             # Day intervention starts
    params_fixed = (c, γ, γ_s, ps, σ, ε, ϕ, intervention_day) # Tuple of fixed parameters

    # **Initial Conditions: S(0), I(0), Is(0), R(0)**
    S0 = 5999
    I0 = 1
    Is0 = 0
    R0 = 0
    initial_conditions = [S0, I0, Is0, R0]

    # **Time Span for the Simulation (in days)**
    tspan = (0, 80)  # Simulate for 80 days

    # Town 1
    # Infected people data (after day 30 intervention is added)
    days_infected = 15:80
    infected_people = [11, 7, 20, 3, 29, 14, 11, 12, 16, 10, 58, 34, 26, 29, 51, 55, 
                        155, 53, 67, 98, 130, 189, 92, 192, 145, 128, 68, 74, 126, 265,
                        154, 207, 299, 273, 190, 152, 276, 408, 267, 462, 352, 385, 221, 
                        420, 544, 329, 440, 427, 369, 606, 416, 546, 475, 617, 593, 352, 
                        337, 473, 673, 653, 523, 602, 551, 686, 556, 600]
    
    # Severely infected people data
    days_severe = 22:80
    Sinfected_people = [0, 0, 1, 2, 5, 5, 5, 2, 9, 4, 22, 0, 15, 48, 38, 57, 9, 18, 20, 
                        0, 41, 15, 35, 36, 27, 38, 24, 40, 34, 57, 18, 29, 63, 66, 119, 
                        76, 95, 28, 109, 136, 119, 104, 121, 93, 147, 129, 130, 161, 133, 
                        136, 138, 139, 181, 181, 218, 183, 167, 164, 219, 220]

    # Town 2
    # Infected people data for town 2
    T2_days_infected = 27:80
    T2_infected_people = [21, 29, 25, 30, 28, 34, 28, 54, 57, 92, 73, 80, 109, 102, 128,
                            135, 163, 150, 211, 196, 233, 247, 283, 286, 332, 371, 390, 404, 
                            467, 529, 598, 641, 704, 702, 788, 856, 854, 955, 995, 1065, 1106, 
                            1159, 1217, 1269, 1298, 1328, 1339, 1383, 1431, 1422, 1414, 1485, 1464, 
                            1480]

    # Severely infected people data
    T2_days_severe = 27:80
    T2_Sinfected_people = [3, 3, 4, 7, 3, 8, 7, 5, 9, 13, 15, 3, 20, 13, 11, 20, 16, 11, 15, 18, 27, 
                            24, 28, 36, 41, 35, 41, 55, 63, 66, 72, 80, 90, 104, 109, 115, 127, 135, 
                            147, 162, 163, 186, 194, 200, 216, 223, 241, 249, 258, 275, 277, 299, 302, 300]

    # Define a Range of β Values
    beta_values = range(0.03, 0.038, length=20)

    # Compute Errors for Each β in the Range
    errors = [compute_error(β, params_fixed, initial_conditions, tspan, days_infected, infected_people) for β in beta_values]

    # Plot Error as a Function of β to Justify Selection
    p_error = plot(beta_values, errors, xlabel="β", ylabel="Sum of Squared Errors", 
                   title="Error vs β", lw=2)
    display(p_error)

    # Find the β Value That Minimizes the Error
    min_error, min_index = findmin(errors)
    best_beta = beta_values[min_index]
    println("Best β value: ", best_beta, " with minimum error: ", min_error)

    # Solve the Model Again with the Best β
    β = best_beta
    params = (c, β, γ, γ_s, ps, σ, ε, ϕ, intervention_day)
    prob = ODEProblem(sirs_model!, initial_conditions, tspan, params)
    sol = solve(prob)
    
    # Plot each variable using best beta
    T1_All_Data_Plot = plot(sol, vars=(1), label="Susceptible", xlabel="Days", ylabel="Population", title="SIRS Model - Susceptible", lw=2)
    T1_All_Data_Plot = plot!(sol, vars=(2), label="Infected", xlabel="Days", ylabel="Population", title="SIRS Model - Infected", lw=2)
    T1_All_Data_Plot = plot!(sol, vars=(3), label="Severely Infected", xlabel="Days", ylabel="Population", title="SIRS Model - Severely Infected", lw=2)
    T1_All_Data_Plot = plot!(sol, vars=(4), label="Recovered", xlabel="Days", ylabel="Population", title="SIRS Model - Recovered", lw=2)
    T1_All_Data_Plot = scatter!(days_infected, infected_people, label="Infected People Data")
    display(T1_All_Data_Plot)

    #Plot just the infected people
    T1_Infected_Data_Plot = plot(sol, vars=(2), label="Infected", xlabel="Days", ylabel="Population", title="SIRS Model - Infected", lw=2)
    T1_Infected_Data_Plot = scatter!(days_infected, infected_people, label="Infected People Data")
    display(T1_Infected_Data_Plot)

    # Extract the infected population (second variable in the solution)
    infected_population = sol[2, :]  # This accesses the second variable (I) at all time points

    # Find the maximum number of infected people
    max_infected = maximum(infected_population)
    println("Maximum number of infected people: ", max_infected)
end