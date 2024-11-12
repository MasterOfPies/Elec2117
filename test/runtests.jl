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

    # Initial Conditions: S(0), I(0), Is(0), R(0)
    S0 = 5999
    I0 = 1
    Is0 = 0
    R0 = 0
    initial_conditions = [S0, I0, Is0, R0]

    # Time Span for the Simulation (in days)
    tspan = (0, 80)

    # Town 1
    # Infected people data (after day 30 intervention is added)
    days_infected = 15:80
    infected_people = [11, 7, 20, 3, 29, 14, 11, 12, 16, 10, 58, 34, 26, 29, 51, 55, 
                        155, 53, 67, 98, 130, 189, 92, 192, 145, 128, 68, 74, 126, 265,
                        154, 207, 299, 273, 190, 152, 276, 408, 267, 462, 352, 385, 221, 
                        420, 544, 329, 440, 427, 369, 606, 416, 546, 475, 617, 593, 352, 
                        337, 473, 673, 653, 523, 602, 551, 686, 556, 600]
    
    # Severely infected people data
    days_severe = 21:80
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

    # Best beta value of infected people
    # Define a range of β values
    beta_values = range(0.03, 0.038, length=20)

    # Compute errors for each β in the range
    errors = [compute_error(β, params_fixed, initial_conditions, tspan, days_infected, infected_people) for β in beta_values]

    # Plot error as a function of β to fustify selection
    p_error = plot(beta_values, errors, xlabel="β", ylabel="Sum of Squared Errors", 
                   title="Error vs β", lw=2)
    display(p_error)

    # Find the β value that minimizes the error
    min_error, min_index = findmin(errors)
    best_beta = beta_values[min_index]

    # Compute error for the severely infected people
    severe_error = [compute_error(best_beta, params_fixed, initial_conditions, tspan, days_severe, Sinfected_people)]
    severe_error_value = severe_error[1]
    println("Best β value: ", best_beta, " with infected error: ", min_error, " with severely infected error: ", severe_error_value)

    # Solve the model again with the best β
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

    # Plot just the infected people
    T1_Infected_Data_Plot = plot(sol, vars=(2), label="Infected", xlabel="Days", ylabel="Population", title="SIRS Model - Infected", lw=2)
    T1_Infected_Data_Plot = scatter!(days_infected, infected_people, label="Infected People Data")
    display(T1_Infected_Data_Plot)

    # Plot just the severely infected people
    T1_SInfected_Data_Plot = plot(sol, vars=(3), label="Severely Infected", xlabel="Days", ylabel="Population", title="SIRS Model - Severely Infected", lw=2)
    T1_SInfected_Data_Plot = scatter!(days_severe, Sinfected_people, label="Severely Infected People Data")
    display(T1_SInfected_Data_Plot)

    # Extract the infected population
    infected_population = sol[2, :]

    # Find the maximum number of infected people
    max_infected = maximum(infected_population)
    println("Maximum number of infected people: ", max_infected)

    Sinfected_population = sol[3, :]
    Smax_infected = maximum(Sinfected_population)
    println("Maximum number of Severely infected people: ", Smax_infected)

    # Finding the value of ϕ that closest aligns with a given β value
    # Define the target β value we want to approach
    target_beta = 0.03463

    # Range of ϕ values to test
    ϕ_values = range(0, 1, length=100)

    # Variable to track the best ϕ and corresponding β value
    best_ϕ = ϕ_values[1]
    closest_beta = beta_values[1]
    min_diff = abs(closest_beta - target_beta)

    # Iterate over ϕ values
    for ϕ in ϕ_values
        # Update params with the new ϕ value
        params_fixed = (c, γ, γ_s, ps, σ, ε, ϕ, intervention_day)

        # Compute errors for each β in the range
        errors = [compute_error(β, params_fixed, initial_conditions, tspan, days_infected, infected_people) for β in beta_values]

    # Find the β value that minimizes the error for this ϕ
    min_error, min_index = findmin(errors)
    best_beta_for_ϕ = beta_values[min_index]
    
    # Check if it is the best β value
    if abs(best_beta_for_ϕ - target_beta) < min_diff
        best_ϕ = ϕ
        closest_beta = best_beta_for_ϕ
        min_diff = abs(best_beta_for_ϕ - target_beta)
    end
    end

    # Print out the results
    println("Optimal ϕ value to achieve β ≈ 0.03463: ")
    println("Closest achieved β value: ", closest_beta, " with ϕ = ", best_ϕ)

    # Finding the best value for second town
    # Define a range of β values
    beta_values = range(0.03, 0.04, length=20)

    # Change params and intial conditiosn for town 2
    # Fixed Parameters for the SIRS model (excluding β)
    c = 8                             # Contact rate
    γ = 1/7                           # Recovery rate for non-severe cases
    γ_s = 1/14                        # Recovery rate for severe cases
    ps = 0.2                          # Proportion of severe cases
    σ = 1/30                          # Resusceptibility rate
    ε = 0.3                           # Intervention efficacy
    ϕ = 0.8                           # Proportion that uses intervention
    intervention_day = 36             # Day intervention starts
    params_fixed = (c, γ, γ_s, ps, σ, ε, ϕ, intervention_day) # Tuple of fixed parameters

    # Initial Conditions: S(0), I(0), Is(0), R(0)
    S0 = 9999
    I0 = 1
    Is0 = 0
    R0 = 0
    initial_conditions = [S0, I0, Is0, R0]

    # Compute errors for each β in the range for Town 2
    errors_T2 = [compute_error(β, params_fixed, initial_conditions, tspan, T2_days_infected, T2_infected_people) for β in beta_values]

    # Plot error as a function of β to identify optimal β for Town 2
    p_error_T2 = plot(beta_values, errors_T2, xlabel="β", ylabel="Error", 
                      title="Error vs β for Town 2", lw=2)
    display(p_error_T2)

    # Find the β value that minimizes the error for Town 2
    min_error_T2, min_index_T2 = findmin(errors_T2)
    best_beta_T2 = beta_values[min_index_T2]
    println("Best β value for Town 2: ", best_beta_T2, " with minimum error: ", min_error_T2)

    # Solve the model with the best β for Town 2
    params_T2 = (c, best_beta_T2, γ, γ_s, ps, σ, ε, ϕ, intervention_day)
    prob_T2 = ODEProblem(sirs_model!, initial_conditions, tspan, params_T2)
    sol_T2 = solve(prob_T2)

    # Plot the infected population for Town 2 using best β
    T2_Infected_Data_Plot = plot(sol_T2, vars=(2), label="Model - Infected", 
                                 xlabel="Days", ylabel="Population", title="Model vs Data for Town 2")
    T2_Infected_Data_Plot = scatter!(T2_days_infected, T2_infected_people, label="Infected People Data")
    display(T2_Infected_Data_Plot)

    # Plot the severely infected population for Town 2 using the best β
    T2_Severe_Infected_Data_Plot = plot(sol_T2, vars=(3), label="Model - Severely Infected", 
    xlabel="Days", ylabel="Population", title="Severely Infected: Model vs Data for Town 2")
    T2_Severe_Infected_Data_Plot = scatter!(T2_days_severe, T2_Sinfected_people, label="Severely Infected People Data")
    display(T2_Severe_Infected_Data_Plot)
end