import openai
import csv
import pandas as pd
import os
from openai import OpenAI
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

client = OpenAI(api_key = 'API')
model="gpt-4-0125-preview"

system_prompt = '''You are assisting me with Suzuki Reaction condition optimizaiton. The task performance is measured by yield, you can only seleted candidates in the given list. '''

prompt_w = ''' 
    The dataset has 309 samples with the combinations of 5 reaction components 3 components(Ligand, Base, Solvent) need to be optimized for given two reactants (6-Chloroquinoline,[5-Methyl-1-(oxan-2-yl)-1H-indazol-4-yl]boronic acid).
    I'm exploring a subset of reaction parameters detailed as: Ligand Name and SMILES, Base SMILES, Solvent SMILES.

    Please suggest 5 of recommendation reaction parameters set(Ligand, Base, Solvent) to initiate a Bayesian Optimization process, in the following candidate list based on your knowledge of ligands solvent base knowledge of Suzuki reaction:
    Ligand:P(t-Bu)3; Triphenylphosphine; Aphos; Tricyclohexylphosphine; P(o-tol)3; Di-(3s,5s,7s)-adamantan-1-yl(butyl)phosphine; Sphos; Ferrocene,1,1'-bis[bis(1,1-dimethylethyl)phosphino]; Xphos; 1,1-BIS(Diphenylphosphino)Ferrocene; Xantphos
    Base:[Na+].[OH-]; OC([O-])=O.[Na+]; [Cs+].[F-]; O=P([O-])([O-])[O-].[K+].[K+].[K+]; [K+].[OH-]; CC([O-])C.[Li+]; CCN(CC)CC
    Solvent: N#CC; C1COCC1; O=CN(C)C; CO
    
    Your response should only contain the results in the following format smiles are split by '|':
    
    Ligand name|Base SMILES|Solvent SMILES
    
    e.g.
    -------------------
    reaction parameters:
    Di-(3s,5s,7s)-adamantan-1-yl(butyl)phosphine|[Na+].[OH-]|N#CC
    Sphos|[Cs+].[F-]|O=CN(C)C
    Ferrocene,1,1'-bis[bis(1,1-dimethylethyl)phosphino]|OC([O-])=O.[Na+]|CO
    ...

    '''


prompt_a = '''The dataset has 309 samples with the combinations of 5 reaction components 3 components(Phosphine Ligand, Base, Solvent) need to be optimized for given two reactants (6-Chloroquinoline,[5-Methyl-1-(oxan-2-yl)-1H-indazol-4-yl]boronic acid).
    I'm exploring a subset of reaction parameters detailed as: Ligand,Base_SMILES,Solvent_SMILES in the following candidate list: 
    Ligand:P(t-Bu)3; Triphenylphosphine; Aphos; Tricyclohexylphosphine; P(o-tol)3; Di-(3s,5s,7s)-adamantan-1-yl(butyl)phosphine; Sphos; Ferrocene,1,1'-bis[bis(1,1-dimethylethyl)phosphino]; Xphos; 1,1-BIS(Diphenylphosphino)Ferrocene; Xantphos
    Base:[Na+].[OH-]; OC([O-])=O.[Na+]; [Cs+].[F-]; O=P([O-])([O-])[O-].[K+].[K+].[K+]; [K+].[OH-]; CC([O-])C.[Li+]; CCN(CC)CC
    Solvent: N#CC; C1COCC1; O=CN(C)C; CO
    Considering you are doing a Bayesian Optimization to find the best reaction parameters to reache highest performance.
    
    Recommend one new reaction parameter set in the candidate list that can achieve the target reaction yield of 100, based on the performance of existing data and your knowledge of Suzuki reaction. 
         
    Your response should only contain the results in the following format:
    external knowledge the phosphine has more influence on the reaction yield comparing to other factors
     
    The recommendation can not be the same to combinations of the existing data
    Your response should only contain the results in the following format smiles are split by '|':
    Ligand name|Base SMILES|Solvent SMILES
    e.g.
    -------------------
    Di-(3s,5s,7s)-adamantan-1-yl(butyl)phosphine|[Na+].[OH-]|N#CC
    existing data:
'''




prompt_s = '''The dataset has 309 samples with the combinations of 5 reaction components 3 components(Phosphine Ligand, Base, Solvent) need to be optimized for given two reactants (6-Chloroquinoline,[5-Methyl-1-(oxan-2-yl)-1H-indazol-4-yl]boronic acid).
    I'm exploring a subset of reaction parameters detailed as: Ligand,Base_SMILES,Solvent_SMILES.
    
    Considering you are doing a Bayesian Optimization to find the best reaction parameters to reache highest performance.
    
    The following are examples of reaction components combination for a suzuki reaction and the corresponding yield performance simulate the yield performance of reaction combination based on existing data:
    
    Your response should only follows the following format: [YIELD1,YIELD2,YIELD3...] count the reaction numbers carefully.
    
'''

def generate_chat_response(template_message):
    """
    Generates a chat response from GPT (e.g., GPT-3.5-turbo) for a given user prompt.
    
    Parameters:
    - prompt (str): The user's input prompt for the chat.
    
    Returns:
    - str: The generated chat response.
    """
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=template_message
    )
    
    # Extracting and returning the generated text
    return response.choices[0].message.content


warmstarting_message = [
            {"role": "system","content": system_prompt},
            
            {"role": "user", "content": prompt_w}
        ]


surrogate_message=[
            {"role": "system","content": system_prompt},
            
            {"role": "user", "content": prompt_s}
        ]


acquisition_message = [
            {"role": "system","content": system_prompt},
            
            {"role": "user", "content": prompt_a}
        ]




Warm_starting_response = generate_chat_response(warmstarting_message)
#warm starting to give some reaction examples

Warm_starting_response = ''' 
Triphenylphosphine|OC([O-])=O.[Na+]|C1COCC1
Aphos|[K+].[OH-]|O=CN(C)C
P(t-Bu)3|CCN(CC)CC|N#CC
Xphos|O=P([O-])([O-])[O-].[K+].[K+].[K+]|CO
1,1-BIS(Diphenylphosphino)Ferrocene|CC([O-])C.[Li+]|C1COCC1'''



def find_unprocessed_reactions_and_yields(data, reactions, sample_size=40):
    # Convert reactions list to a set of tuples for faster comparison
    processed_reactions_set = set(tuple(reaction) for reaction in reactions)
    
    # Create a set of all unique combinations in the data DataFrame
    all_reactions_set = set(zip(data['Ligand_Names'], data['Base_SMILES'], data['Solvent_SMILES']))
    
    # Find the difference between all reactions and processed reactions
    unprocessed_reactions = all_reactions_set - processed_reactions_set
    
    # Initialize a DataFrame for unprocessed reactions and their yields
    unprocessed_data = pd.DataFrame(columns=['Ligand_Names', 'Base_SMILES', 'Solvent_SMILES', 'yield'])
    
    # Iterate through unprocessed reactions to extract their yield
    for reaction in unprocessed_reactions:
        ligand, base, solvent = reaction
        filtered_data = data[
            (data['Ligand_Names'] == ligand) &
            (data['Base_SMILES'] == base) &
            (data['Solvent_SMILES'] == solvent)
        ]
        
        # If there are any matches, append their data to the unprocessed_data DataFrame
        if not filtered_data.empty:
            unprocessed_data = unprocessed_data.append(filtered_data, ignore_index=True)
    
    # Sample 100 data points from the unprocessed_data DataFrame if it has enough rows
    if len(unprocessed_data) > sample_size:
        sampled_data = unprocessed_data.sample(n=sample_size, random_state=42)
    else:
        sampled_data = unprocessed_data
    
    # Extract the sampled unprocessed reactions and yields
    sampled_unprocessed_reactions = list(zip(sampled_data['Ligand_Names'], sampled_data['Base_SMILES'], sampled_data['Solvent_SMILES']))
    sampled_yields = sampled_data['yield'].tolist()
    
    return sampled_unprocessed_reactions, sampled_yields
    

def load_experiment_data(file_path):
    return pd.read_csv(file_path) 




if __name__ == "__main__":


  file_path = 'experiment_index.csv'
  data = load_experiment_data(file_path)

  reactions = []
  starting_reaction = []
  yield_results = []


  for line in Warm_starting_response.split('\n'):
     reaction = line.split('|')
     if len(reaction)==3:
        reactions.append(reaction)
        

  for ligand, base, solvent in reactions:
     # Filter the DataFrame for each set of components
     filtered_data = data[
        (data['Ligand_Names'] == ligand) &
        (data['Base_SMILES'] == base) &
        (data['Solvent_SMILES'] == solvent)
     ]
     
     # Check if there are any matches, and extract the yield information
     if not filtered_data.empty:
        yield_info = filtered_data[['Ligand_Names','Base_SMILES', 'Solvent_SMILES', 'yield']].values.tolist()
        starting_reaction.extend(yield_info)
        
  new_str = ''

  for result in starting_reaction:
     yield_results.append(result[3])
     string = '|'.join([str(x) for x in result])
     new_str+=string
     new_str+='\n'
    
  prompt_a =  prompt_a+'\n'+new_str 
  yield_values = 0
  
  new_yield = []
  MAE = []
  existing_reaction = []
  n=1
  while(yield_values<90):
     a_response = generate_chat_response(acquisition_message)
     reactions.append(a_response.split('|'))
     ligand, base, solvent = a_response.split('|')
     print('new recommended reaction parameters: ',ligand, base, solvent)
     filtered_data = data[
     (data['Ligand_Names'] == ligand) &
     (data['Base_SMILES'] == base) &
     (data['Solvent_SMILES'] == solvent)  ]

     yield_values = float(filtered_data['yield'].iloc[0])
     yield_results.append(yield_values)
     new_yield.append(yield_values)
     unprocessed,true_yield = find_unprocessed_reactions_and_yields(data,reactions)
    
    
     for i in range(len(reactions)):
    
       if i not in existing_reaction:
           surrogate_message.append({"role": "user", "content": "{}".format('|'.join(reactions[i]))})
           surrogate_message.append({"role": "assistant", "content": "[{}]".format(str(yield_results[i]))})
           existing_reaction.append(i)
    
     def chunk_list(data, chunk_size):
        """Yield successive chunk_size chunks from data."""
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
     print(surrogate_message)
     def prepare_surrogate_messages(unprocessed, chunk_size=20):
        surrogate_responses = []
        for chunk in chunk_list(unprocessed, chunk_size):
           # Prepare the message for the current chunk of reactions
           message_content = "20 smiles:\n{}".format('\n'.join(['|'.join(x) for x in chunk]))
           # Append the prepared message directly to the surrogate_messages list
           surrogate_message.append({"role": "user", "content": message_content})
           response = generate_chat_response(surrogate_message)
           surrogate_message.remove({"role": "user", "content": message_content})
           real_list = ast.literal_eval(response)
           surrogate_responses.extend(real_list)

        return surrogate_responses
    
     #surrogate_message.append({"role": "user", "content": "Predict the yield of reaction:{} \n Your response should only contain the results in the following format: [YIELD1,YIELD2,YIELD3...]".format('\n'.join(['|'.join(x) for x in unprocessed]))})
     #print(surrogate_message,'__________________________')
     s_response = prepare_surrogate_messages(unprocessed)
     print(s_response)
    
     mae = mean_absolute_error(true_yield, s_response)
     MAE.append(mae)
     print('MAE',mae)
     print('new_yield',yield_values)
    
     prompt_a+='\n{}|{}'.format(a_response,yield_values)


    
     n+=1
    
     if n==30:
        break

  print(new_yield)
  print(MAE)
  if 1:
    
     # Generating the iteration numbers starting from 1
     iterations = list(range(1, len(new_yield) + 1))

     # Plotting Yield vs. Iterations
     plt.figure(figsize=(10, 5))
     plt.plot(iterations, new_yield, '-x', color='blue', linewidth=2, markersize=8)
     plt.xlabel('Iteration Number')
     plt.ylabel('Yield')
     plt.title('Yield Over Iterations')
     plt.grid(True, linestyle='--', linewidth=0.5)
     plt.show()
    
     plt.figure(figsize=(10, 5))
     plt.plot(iterations, MAE, '-x', color='blue', linewidth=2, markersize=8)
     plt.xlabel('Iteration Number')
     plt.ylabel('MAE')
     plt.title('MAE Over Iterations')
     plt.grid(True, linestyle='--', linewidth=0.5)
     plt.show()








