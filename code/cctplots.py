import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

def load_plant_knowledge_data(csv_file_path=None):
    """
    Loads the plant knowledge dataset from a CSV file
    
    Parameters:
    csv_file_path (str, optional): Path to the CSV file. If None, uses default location.
    """
    try:
        # If no path provided, construct path to project data directory
        if csv_file_path is None:
            # Get directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the project root
            project_root = os.path.dirname(script_dir)
            # Path to data directory
            csv_file_path = os.path.join(project_root, "data", "plant_knowledge.csv")
        
        print(f"Attempting to load data from: {csv_file_path}")
        
        df = pd.read_csv(csv_file_path)
        
        # Drop "Informant" column if it exists
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

    N = data.shape[0]  # Number of informants
    M = data.shape[1]  # Number of questions

    with pm.Model() as model:
        # Priors
        # For informant competence (D), uniform prior between 0.5 and 1.
        D = pm.Uniform("D", lower=0.5, upper=1, shape=N)
        # For consensus answers (Z), bernoulli prior with p=0.5.
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        # Likelihood
        D_reshaped = D[:, None]  # Shape (N, 1)
        Z_reshaped = Z[None, :]  # Shape (1, M)

        # Calculate the probability p_ij for each informant i and question j
        p = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D_reshaped)

        # Bernoulli likelihood function.
        pm.Bernoulli("X", p=p, observed=data)

    return model


def analyze_results(trace, data, save_plots=True, plots_dir="plots"):
    """
    Analyzes the results of the PyMC CCT model.
    
    Parameters:
    trace: PyMC trace object
    data: The original data used for the model
    save_plots: Boolean indicating whether to save plots to files
    plots_dir: Directory to save plots (will be created if it doesn't exist)
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
    
    # -------------------------------------------
    # PAIR PLOT HANDLING - MODIFIED TO FIX ISSUE
    # -------------------------------------------
    
    # For D values, create a more manageable subset for the pair plot
    # Select only a few informants to avoid exceeding the subplot limit
    max_informants_to_plot = 6  # Adjust based on your needs
    D_indices_to_plot = list(range(min(N, max_informants_to_plot)))
    
    # For Z values, also select a manageable subset
    max_questions_to_plot = 6  # Adjust based on your needs
    Z_indices_to_plot = list(range(min(M, max_questions_to_plot)))
    
    # Prepare variable names for the pair plot
    D_vars = [f'D[{i}]' for i in D_indices_to_plot]
    Z_vars = [f'Z[{j}]' for j in Z_indices_to_plot]
    
    # Create separate pair plots for D and Z
    print("\nCreating pair plot for informant competence (D)...")
    if D_vars:
        fig_pair_D = az.plot_pair(trace, var_names=D_vars, figsize=(12, 10))
        plt.suptitle("Pair Plot of Informant Competence (D)", fontsize=16)
        if save_plots:
            plt.savefig(os.path.join(plots_path, "pairplot_D.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\nCreating pair plot for consensus answers (Z)...")
    if Z_vars:
        fig_pair_Z = az.plot_pair(trace, var_names=Z_vars, figsize=(12, 10))
        plt.suptitle("Pair Plot of Consensus Answers (Z)", fontsize=16)
        if save_plots:
            plt.savefig(os.path.join(plots_path, "pairplot_Z.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    # If you have few enough variables, you can also create a combined plot
    if len(D_vars) + len(Z_vars) <= 12:  # A reasonable limit
        combined_vars = D_vars[:3] + Z_vars[:3]  # Take first few of each
        fig_pair_combined = az.plot_pair(trace, var_names=combined_vars, figsize=(12, 10))
        plt.suptitle("Combined Pair Plot (Selected D and Z)", fontsize=16)
        if save_plots:
            plt.savefig(os.path.join(plots_path, "pairplot_combined.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    # -------------------------------------------
    # END OF PAIR PLOT HANDLING
    # -------------------------------------------

    # Estimate Informant Competence
    print("\nInformant Competence (D):")
    D_mean = trace.posterior["D"].mean(dim=("chain", "draw")).values
    for i, d in enumerate(D_mean):
        print(f"Informant {i+1}: {d:.3f}")

    # Visualize posterior distributions of D
    fig_d = az.plot_posterior(trace, var_names=['D'])
    plt.title("Posterior Distributions of Informant Competence (D)")
    if save_plots:
        plt.savefig(os.path.join(plots_path, "posterior_D.png"), dpi=300)
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
        plt.savefig(os.path.join(plots_path, "posterior_Z.png"), dpi=300)
    plt.show()

    # Compare with Naive Aggregation (Majority Vote)
    majority_vote = np.round(data.mean(axis=0))
    print("\nMajority Vote Answer Key:")
    print(majority_vote.astype(int))

    print("\nComparison with CCT:")
    print("Differences between majority vote and CCT estimates may occur because CCT accounts for varying informant competence, while majority vote treats all informants equally. CCT down-weights the answers of less competent informants, leading to potentially more accurate estimates of the true consensus.")
    
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
    
    print("\nAnalysis complete! Check the plots directory for visualization files.")

if __name__ == "__main__":
    main()
