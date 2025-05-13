import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os  # Import the os module for path manipulation

def load_plant_knowledge_data(csv_file_path="plant_knowledge.csv"):
    """
    Loads the plant knowledge dataset from a CSV file
    """
    try:
        # Use os.path.abspath to get the absolute path
        abs_file_path = os.path.abspath(csv_file_path)
        df = pd.read_csv(abs_file_path)

        # Drop "Informant" column
        df = df.drop(columns=["Informant"])
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {abs_file_path}")
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

def analyze_results(trace, data):
    """
    Analyzes the results of the PyMC CCT model.
    """
    
    N = data.shape[0]
    M = data.shape[1]

    # Convergence diagnostics
    print("Convergence Diagnostics:")
    summary = az.summary(trace)
    print(summary)
    az.plot_pair(trace, var_names=['D', 'Z'])
    plt.show()

    # Estimate Informant Competence
    print("\nInformant Competence (D):")
    D_mean = trace.posterior["D"].mean(dim=("chain", "draw")).values
    for i, d in enumerate(D_mean):
        print(f"Informant {i+1}: {d:.3f}")

    # Visualize posterior distributions of D
    az.plot_posterior(trace, var_names=['D'])
    plt.title("Posterior Distributions of Informant Competence (D)")
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
    az.plot_posterior(trace, var_names=['Z'])
    plt.title("Posterior Probabilities of Consensus Answers (Z)")
    plt.show()

    # Compare with Naive Aggregation (Majority Vote)
    majority_vote = np.round(data.mean(axis=0))
    print("\nMajority Vote Answer Key:")
    print(majority_vote.astype(int))

    print("\nComparison with CCT:")
    print("Differences between majority vote and CCT estimates may occur because CCT accounts for varying informant competence, while majority vote treats all informants equally. CCT down-weights the answers of less competent informants, leading to potentially more accurate estimates of the true consensus.")

def main():
    """
    Main function to load data, run the CCT model, and analyze results.
    """
    # Load the data
    plant_knowledge_df = load_plant_knowledge_data()
    if plant_knowledge_df is None:
        print("Error: Failed to load data. Exiting.")
        return

    # Convert the Pandas DataFrame to a NumPy array for PyMC
    plant_knowledge_data = plant_knowledge_df.values

    # Implement the model in PyMC
    model = cct_model(plant_knowledge_data)

    # Perform inference
    with model:
        trace = pm.sample(draws=2000, chains=4, tune=1000)

    # Analyze the results
    analyze_results(trace, plant_knowledge_data)

if __name__ == "__main__":
    main()

