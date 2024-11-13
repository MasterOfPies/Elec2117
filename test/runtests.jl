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
    severe_error = [compute_error_severe(best_beta, params_fixed, initial_conditions, tspan, days_severe, Sinfected_people)]
    severe_error_value = severe_error[1]
    println("Best β value: ", round(best_beta, digits=3), " with infected error: ", round(min_error, digits=3), " with severely infected error: ", round(severe_error_value, digits=3))

    # Finding percentage error
    average = mean(infected_people)
    percentage_error = (min_error/average)

    average_severe = mean(Sinfected_people)
    percentage_error_severe = (severe_error_value/average_severe)

    overall_percentage_error = (percentage_error + percentage_error_severe)/2

    println("Percentage Error: ", round(100 * overall_percentage_error, digits=3), "%")
    println("β = ", round(best_beta, digits=3), " ± ", round(best_beta * (overall_percentage_error), digits=5))
    println("")

    # Solve the model again with the best β
    β = best_beta
    params = (c, β, γ, γ_s, ps, σ, ε, ϕ, intervention_day)
    prob = ODEProblem(sirs_model!, initial_conditions, tspan, params)
    sol = solve(prob)

    # Solve model again with no intervention
    β = best_beta
    params = (c, β, γ, γ_s, ps, σ, ε, ϕ, Inf)
    prob_no_int = ODEProblem(sirs_model!, initial_conditions, tspan, params)
    sol_no_int = solve(prob_no_int)


    # Plot each variable using best beta
    T1_All_Data_Plot = plot(sol, idxs=(1), label="Susceptible", xlabel="Days", ylabel="Population", title="SIRS Model - Susceptible", lw=2)
    T1_All_Data_Plot = plot!(sol, idxs=(2), label="Infected", xlabel="Days", ylabel="Population", title="SIRS Model - Infected", lw=2)
    T1_All_Data_Plot = plot!(sol, idxs=(3), label="Severely Infected", xlabel="Days", ylabel="Population", title="SIRS Model - Severely Infected", lw=2)
    T1_All_Data_Plot = plot!(sol, idxs=(4), label="Recovered", xlabel="Days", ylabel="Population", title="SIRS Model", lw=2)
    T1_All_Data_Plot = scatter!(days_infected, infected_people, label="Infected People Data")
    display(T1_All_Data_Plot)

    # Plot of infected people
    T1_Infected_Data_Plot = plot(sol, idxs=(2), label="Infected", xlabel="Days", ylabel="Population", title="SIRS Model - Infected with Error Bounds", lw=2)
    T1_Infected_Data_Plot = plot!(sol_no_int, idxs=(2), label="Infected No Intervention", xlabel="Days", ylabel="Population", title="SIRS Model - Infected with Error Bounds", lw=2)
    infected_with_error = sol[2, :] .* overall_percentage_error
    T1_Infected_Data_Plot = plot!(sol.t, sol[2, :], ribbon=infected_with_error, fillalpha=0.2, label="Error Bounds")
    scatter!(T1_Infected_Data_Plot, days_infected, infected_people, label="Infected People Data")
    display(T1_Infected_Data_Plot)

    # Plot just the severely infected people
    T1_SInfected_Data_Plot = plot(sol, idxs=(3), label="Severely Infected", xlabel="Days", ylabel="Population", title="SIRS Model - Severely Infected with Error Bounds", lw=2)
    T1_SInfected_Data_Plot = plot!(sol_no_int, idxs=(3), label="Severely Infected No Intervention", xlabel="Days", ylabel="Population", title="SIRS Model - Severely Infected with Error Bounds", lw=2)
    Sinfected_with_error = sol[3, :] .* overall_percentage_error
    T1_SInfected_Data_Plot = plot!(sol.t, sol[3, :], ribbon=Sinfected_with_error, fillalpha=0.2, label="Error Bounds")
    scatter!(T1_SInfected_Data_Plot, days_severe, Sinfected_people, label="Severely Infected People Data")
    display(T1_SInfected_Data_Plot)

    # Extract the infected population
    infected_population = sol[2, :]
    infected_population_no_int = sol_no_int[2, :]

    # Find the maximum number of infected people
    max_infected = maximum(infected_population)
    max_infected_no_int = maximum(infected_population_no_int)
    println("Maximum number of infected people using best β value: ", round(max_infected, digits=3))
    println("Maximum number of infected people with no intervention: ", round(max_infected_no_int, digits=3))
    println("With an Error of ±", round(max_infected * overall_percentage_error, digits=3))
    println("")

    # Finds the max number of severley infected people and for other beta values
    Sinfected_population = sol[3, :]
    Sinfected_population_no_int = sol_no_int[3, :]

    Smax_infected = maximum(Sinfected_population)
    Smax_infected_no_int = maximum(Sinfected_population_no_int)

    println("Maximum number of severely infected people using best β value: ", round(Smax_infected, digits=3))
    println("Maximum number of severely infected people with no intervention: ", round(Smax_infected_no_int, digits=3))
    println("With an Error of ±", round(Smax_infected * overall_percentage_error, digits=3))
    println("")

    # Assume that this is the best β value
    best_best = β

    # Range of ϕ values to test
    ϕ_values = range(0, 1, length=1000)

    # Variables to track the optimal ϕ and the minimum error
    optimal_ϕ = 0
    min_error = Inf
    errors_ϕ = Float64[]

    # Iterate over each ϕ value to find the one that minimizes the error
    for ϕ in ϕ_values
        # Update params with the fixed best_beta and current ϕ value
        params_fixed = (c, γ, γ_s, ps, σ, ε, ϕ, intervention_day)

        # Compute the error for the current ϕ
        error = compute_error(best_beta, params_fixed, initial_conditions, tspan, days_infected, infected_people)

        push!(errors_ϕ, error)

        # Check if this ϕ produces a smaller error
        if error < min_error
            optimal_ϕ = ϕ
            min_error = error
        end
    end

    # Plot error as a function of ϕ
    ϕ_errors_plot = plot(ϕ_values, errors_ϕ, xlabel="ϕ", ylabel="Error", title="Error vs ϕ", lw=2)
    display(ϕ_errors_plot)

    # Print out the optimal ϕ value and the corresponding error
    println("Optimal ϕ value: ", round(optimal_ϕ, digits=3))
    println("Minimum error with β = ", round(best_beta, digits=5))
    println("Square Root Mean Error: ", round(min_error, digits=3))
    println("")

    # Finding the best value for second town

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

    # Definition of variables
    start_days = 0:30
    best_start_day = start_days[1]
    overall_min_error = Inf
    overall_best_beta = 0 
    best_beta_T2 = 0
    errors_T2 = Inf
    min_error_T2 = Inf
    min_index_T2 = 0

    # Iterate over each possible start day
    for start_day in start_days
    # Adjust the time span to start from the current start_day
    tspan = (start_day, 80)

    # Define a range of β values
    beta_values = range(0.035, 0.045, length=100)

    # Compute errors for each β in the range for Town 2
    errors_T2 = [compute_error(β, params_fixed, initial_conditions, tspan, T2_days_infected, T2_infected_people) for β in beta_values]

    # Find the β value that minimizes the error for Town 2
    min_error_T2, min_index_T2 = findmin(errors_T2)
    best_beta_T2 = beta_values[min_index_T2]

    if min_error_T2 < overall_min_error
        overall_min_error = min_error_T2
        best_start_day = start_day
        overall_best_beta = best_beta_T2
    end

    end

    # Find the start day errors using calculated β
    start_day_errors = Float64[]

    # Iterate over each possible start day to calculate the error
    for start_day in start_days
        # Adjust the time span to start from the current start_day
        tspan = (start_day, 80)

        # Calculate the error for the best β value with this start day
        error_for_start_day = compute_error(overall_best_beta, params_fixed, initial_conditions, tspan, T2_days_infected, T2_infected_people)

        # Append the calculated error to the start_day_errors array
        push!(start_day_errors, error_for_start_day)
    end

    # Plot the error as a function of start day
    start_error_plot = plot(start_days, start_day_errors, xlabel="Start Day", ylabel="Error", title="Error vs Start Day in Town 2", lw=2)
    display(start_error_plot)

    # Set the new tspan
    tspan = (best_start_day, 80)

    # Solve the model with the best β for Town 2
    params_T2 = (c, overall_best_beta, γ, γ_s, ps, σ, ε, ϕ, intervention_day)
    prob_T2 = ODEProblem(sirs_model!, initial_conditions, tspan, params_T2)
    sol_T2 = solve(prob_T2)
    errors_T2 = [compute_error(β, params_fixed, initial_conditions, tspan, T2_days_infected, T2_infected_people) for β in beta_values]
    Serrors_T2 = compute_error_severe(overall_best_beta, params_fixed, initial_conditions, tspan, T2_days_severe, T2_Sinfected_people)
    
    # Plot error as a function of β to identify optimal β for Town 2
    p_error_T2 = plot(beta_values, errors_T2, xlabel="β", ylabel="Error", 
                      title="Error vs β for Town 2", lw=2)
    display(p_error_T2)

    # Calculate percentage error for infected individuals in Town 2
    average_T2_infected = mean(T2_infected_people)
    percentage_error_T2_infected = overall_min_error / average_T2_infected

    # Calculate percentage error for severely infected individuals in Town 2
    average_T2_severe = mean(T2_Sinfected_people)
    percentage_error_T2_severe = Serrors_T2 / average_T2_severe

    # Calculate the overall percentage error as the average of the two errors
    overall_percentage_error_T2 = (percentage_error_T2_infected + percentage_error_T2_severe) / 2

    println("Town 2")
    println("Best β value for Town 2: ", round(overall_best_beta, digits=3))
    println("Infected individuals error: ", round(overall_min_error, digits=3))
    println("Severely infected individuals error: ", Serrors_T2)
    println("Infected Percentage Error for Town 2: ", round(100 * percentage_error_T2_infected, digits=3), "%")
    println("Severely Infected Percentage Error for Town 2: ", round(100 * percentage_error_T2_severe, digits=3), "%")
    println("Overall Percentage Error for Town 2: ", round(100 * overall_percentage_error_T2, digits=3), "%")
    println("Optimal start day for the model in Town 2: ", best_start_day)
    
    # Plot the infected population for Town 2 using best β
    T2_Infected_Data_Plot = plot(sol_T2, idxs=(2), label="Model - Infected", xlabel="Days", ylabel="Population", title="Model vs Data for Town 2 with Error Bounds", lw=2)
    T2_infected_with_error = sol_T2[2, :] .* overall_percentage_error
    T2_Infected_Data_Plot = plot!(sol_T2.t, sol_T2[2, :], ribbon=T2_infected_with_error, fillalpha=0.2, label="Error Bounds")
    scatter!(T2_Infected_Data_Plot, T2_days_infected, T2_infected_people, label="Infected People Data")
    display(T2_Infected_Data_Plot)

    # Plot the severely infected population for Town 2 using the best β
    T2_Severe_Infected_Data_Plot = plot(sol_T2, idxs=(3), label="Model - Severely Infected", xlabel="Days", ylabel="Population", title="Severely Infected: Model vs Data for Town 2 with Error Bounds", lw=2)
    T2_Sinfected_with_error = sol_T2[3, :] .* overall_percentage_error
    T2_Severe_Infected_Data_Plot = plot!(sol_T2.t, sol_T2[3, :], ribbon=T2_Sinfected_with_error, fillalpha=0.2, label="Error Bounds")
    scatter!(T2_Severe_Infected_Data_Plot, T2_days_severe, T2_Sinfected_people, label="Severely Infected People Data")
    display(T2_Severe_Infected_Data_Plot)

    # Start day errors
    plot(start_days, start_day_errors, xlabel="Start Day", ylabel="Error", title="Error vs Start Day in Town 2", lw=2)

    # Plot if there was always intervention
    params_T2 = (c, overall_best_beta, γ, γ_s, ps, σ, ϵ, ϕ, intervention_day)
    prob_T2 = ODEProblem(sirs_model!, initial_conditions, tspan, params_T2)
    sol_T2 = solve(prob_T2)
    T2_Infected_Data_Plot = plot(sol_T2, idxs=(2), label="Model - Infected", xlabel="Days", ylabel="Population", title="Model for T2 with Intervention at Day 1", lw=2)
    display(T2_Infected_Data_Plot)
    println("")
end