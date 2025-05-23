import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

def load_plant_knowledge_data(csv_file_path=None):
    """
    Loads the plant knowledge dataset from a CSV file
    """
    try:
        if csv_file_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            csv_file_path = os.path.join(project_root, "data", "plant_knowledge.csv")
        
        print(f"Attempting to load data from: {csv_file_path}")
        
        df = pd.read_csv(csv_file_path)
        
        # Drop "Informant" column
        if "Informant" in df.columns:
            df = df.drop(columns=["Informant"])
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def cct_model(data):
    """
    Implements the Cultural Consensus Theory (CCT) model using PyMC.
    """
   # Gemini used to aid with creating CCT model code and analysis code
   
    N = data.shape[0]  # informants
    M = data.shape[1]  # questions

    with pm.Model() as model:
        # Priors
        D = pm.Uniform("D", lower=0.5, upper=1, shape=N)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        # Likelihood
        D_reshaped = D[:, None]  # Shape (N, 1)
        Z_reshaped = Z[None, :]  # Shape (1, M)

        # p_ij Equation
        p = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D_reshaped)

        # Bernoulli likelihood function
        pm.Bernoulli("X", p=p, observed=data)

    return model


def analyze_results(trace, data, save_plots=True, plots_dir="plots"):
    """
    Analyzes the results of the PyMC CCT model.
    """
    N = data.shape[0]
    M = data.shape[1]

    # Create plots directory if saving plots
    if save_plots:
        # Create plots directory relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        plots_path = os.path.join(project_root, plots_dir)
        
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
            print(f"Created plots directory: {plots_path}")
        else:
            print(f"Plots will be saved to: {plots_path}")

    # Convergence diagnostics
    print("Convergence Diagnostics:")
    summary = az.summary(trace)
    print(summary)
    
    has_d = "D" in trace.posterior.data_vars
    has_z = "Z" in trace.posterior.data_vars
        
    # Extract posterior samples for D and Z
    d_samples = trace.posterior["D"].values.reshape(-1, N) 
    z_samples = trace.posterior["Z"].values.reshape(-1, M) 
    
    # Estimate Informant Competence
    print("\nInformant Competence (D):")
    D_mean = trace.posterior["D"].mean(dim=("chain", "draw")).values
    for i, d in enumerate(D_mean):
        print(f"Informant {i+1}: {d:.3f}")

    # Visualize posterior distributions of D
    fig_d = az.plot_posterior(trace, var_names=['D'])
    plt.title("Posterior Distributions of Informant Competence (D)")
    if save_plots:
        plt.savefig(os.path.join(plots_path, "posterior_competence_D.png"), dpi=300)
    plt.show()

    # Identify most and least competent informants
    most_competent = np.argmax(D_mean) + 1
    least_competent = np.argmin(D_mean) + 1
    print(f"\nMost Competent Informant: {most_competent}")
    print(f"Least Competent Informant: {least_competent}")

    # Estimate Consensus Answers
    print("\nConsensus Answers (Z):")
    Z_mean = trace.posterior["Z"].mean(dim=("chain", "draw")).values
    for j, z in enumerate(Z_mean):
        print(f"Question {j+1}: P(Z_{j+1} = 1) = {z:.3f}")

    # Determine most likely consensus answer key
    consensus_answers = np.round(Z_mean)  # Round to 0 or 1
    print("\nMost Likely Consensus Answer Key (Z):")
    print(consensus_answers.astype(int))

    # Visualize posterior probabilities for Z
    fig_z = az.plot_posterior(trace, var_names=['Z'])
    plt.title("Posterior Probabilities of Consensus Answers (Z)")
    if save_plots:
        plt.savefig(os.path.join(plots_path, "posterior_consensus_Z.png"), dpi=300)
    plt.show()

    # Compare with Naive Aggregation (Majority Vote)
    majority_vote = np.round(data.mean(axis=0))
    print("\nMajority Vote Answer Key:")
    print(majority_vote.astype(int))
    
    if save_plots:
        print(f"\nAll plots have been saved to the '{plots_path}' directory.")


def main():
    """
    Main function to load data, run the CCT model, and analyze results.
    """
    plant_knowledge_df = load_plant_knowledge_data()
    if plant_knowledge_df is None:
        print("Error: Failed to load data. Exiting.")
        return
       
    plant_knowledge_data = plant_knowledge_df.values

    model = cct_model(plant_knowledge_data)

    with model:
        trace = pm.sample(draws=2000, chains=4, tune=1000, return_inferencedata=True)

    analyze_results(trace, plant_knowledge_data, save_plots=True, plots_dir="plots")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
   
