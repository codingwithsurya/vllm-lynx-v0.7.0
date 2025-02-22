"""Defining the MixtralLogitStore class to store the router logits computed in forward pass of MixtralMoE for each token in the sequence."""
from typing import List, Optional, Tuple

import numpy as np
import os
import torch
import csv
# import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vllm.config import ModelConfig
import torch.nn.functional as F
# from vllm.engine.cdf_sketch import CDFSketch
from vllm.sequence import SequenceGroupMetadata
from datetime import datetime
import pickle
# import plotly.express as px



# define a constant for plotting dir
PLOTTING_DIR = "./visualizing_experts"

def if_enabled(func):
    def wrapper(self, *args, **kwargs):
        if not self.disable and self.profile_complete:
            return func(self, *args, **kwargs)

    return wrapper

def if_enabled_mini(func):
    def wrapper(self, *args, **kwargs):
        if not self.disable_mini:
            return func(self, *args, **kwargs)
        
    return wrapper


class MixtralLogitStore:
    _instance = None

    """This code dumps the router logits computed in forward pass of MixtralMoE for each token in he sequence."""
    def __init__(self, model_config: ModelConfig):
        self.logit_metrics = {}
        self.router_logit_store = []
        self.modified_logit_store = []
        self.expert_counts = []
        self.batch_idx = 0
        self.profile_complete = False
        self.is_prefill = True
        print("The MixtralLogitStore is created and prefill is set to true: ", self.is_prefill)
        self.experts_dropped = 0
        self.logit_logging_frequency = model_config.logit_logging_frequency
        self.record_next_batch = False
        
        # if wandb.run is None:
        #     wandb.init(project="prowl",
        #             group="visualizing experts",
        #             config=model_config.__dict__,
        #             )
        # else:
        #     self.run = wandb.run
        
        # Get the current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Define the full path including the base directory and the new subdirectory with date and time
        self.full_path = os.path.join(PLOTTING_DIR, f"plots_{current_datetime}")
        # Use os.makedirs to create the directory, including all necessary parent directories
        os.makedirs(self.full_path, exist_ok=True)
        # self.disable = model_config.disable_logit_logging
        self.disable = True
        self.disable_mini = False
        
    @classmethod
    def get_instance(cls):
        assert cls._instance is not None, "Instance not created. Use get_or_create_instance() to create an instance."
        return cls._instance
        
        
    @classmethod
    def create_instance(cls, model_config: ModelConfig):
        if cls._instance is None:
            cls._instance = MixtralLogitStore(model_config)
        return cls._instance
    
    #### Important functions which are required for functionality of expert dropping, never comment these out
    
    def mark_profiling_done(self):
        self.profile_complete = True
        print("The profiling is done and the flag is set for MixtralLogitStore")

    def on_batch_start_worker(self, seq_group_metadata_list: Optional[List[SequenceGroupMetadata]]):
        # print("The batch is starting and we are checking if the profiling is done, the profile_complete flag is: ", self.profile_complete, "the prefilled flag is: ", self.is_prefill)
        if not seq_group_metadata_list:
            return
        if self.profile_complete:
            # print("The profiling is already done")
            self.is_prefill = all([seq_group_metadata.is_prompt for seq_group_metadata in seq_group_metadata_list])
            # if self.is_prefill:
            #     print("We encountered a prefill batch and setting the flag for mixtral logit store")
            # else:
            #     print("We encountered a decode batch and setting the flag for mixtral logit store")  
        
    def on_batch_end_worker(self):
        self.batch_idx += 1

    ##############################################################

    def dump_router_logits_per_batch(self, router_logits, layer_idx):
        if router_logits.is_cuda:
            router_logits = router_logits.cpu()
        
        self.router_logit_store.append((self.batch_idx, layer_idx, router_logits))

        # Dump to pickle file after each append
        save_dir = 'router_logits_instruct'
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'router_logits_batch_{self.batch_idx}_layer_{layer_idx}.pkl')
        
        with open(filename, 'wb') as f:
            pickle.dump((self.batch_idx, layer_idx, router_logits), f)
        
        # print(f"Saved router logits for batch {self.batch_idx}, layer {layer_idx} to {filename}")

    def log_unique_experts_per_layer(self, unique_experts, layer_idx):
        if "unique_experts" not in self.logit_metrics:
            self.logit_metrics["unique_experts"] = {}
        if layer_idx not in self.logit_metrics["unique_experts"]:
            self.logit_metrics["unique_experts"][layer_idx] = []

        self.logit_metrics["unique_experts"][layer_idx].append(unique_experts)

    def log_conf_percentage_per_layer(self, conf_percentage, layer_idx):
        if "conf_percentage" not in self.logit_metrics:
            self.logit_metrics["conf_percentage"] = {}
        if layer_idx not in self.logit_metrics["conf_percentage"]:
            self.logit_metrics["conf_percentage"][layer_idx] = []
        
        self.logit_metrics["conf_percentage"][layer_idx].append(conf_percentage)

    def log_experts_dropped_per_layer(self, experts_dropped, layer_idx):
        if "experts_dropped" not in self.logit_metrics:
            self.logit_metrics["experts_dropped"] = {}
        if layer_idx not in self.logit_metrics["experts_dropped"]:
            self.logit_metrics["experts_dropped"][layer_idx] = []
        
        self.logit_metrics["experts_dropped"][layer_idx].append(experts_dropped)
    
    def plot_conf_percentage_per_layer(self):
        # print("Plotting confidence percentage per layer")
        if "conf_percentage" not in self.logit_metrics:
            raise KeyError("No confidence percentage logged.")
        
        # Prepare data for plotting
        data = []
        for layer_idx, percentages in self.logit_metrics["conf_percentage"].items():
            data.extend([(layer_idx, p) for p in percentages])
        
        df = pd.DataFrame(data, columns=['Layer', 'Confidence Percentage'])
        
        # Create the plot
        # plt.figure(figsize=(12, 6))
        # sns.violinplot(x='Layer', y='Confidence Percentage', data=df)
        
        # plt.title('Confidence Percentage Distribution per Layer')
        # plt.xlabel('Layer Index')
        # plt.ylabel('Confidence Percentage')
        
        # # Improve x-axis labels if there are many layers
        # if len(self.logit_metrics["conf_percentage"]) > 10:
        #     plt.xticks(rotation=45)
        
        # plt.tight_layout()
        # wandb.log({"Confidence Percentage per Layer": wandb.Image(plt)})
        # plt.close()

    def plot_experts_dropped_per_layer(self):
        # print("Plotting confidence percentage per layer")
        if "experts_dropped" not in self.logit_metrics:
            raise KeyError("No experts dropped in this layer.")
        
        # Prepare data for plotting
        data = []
        for layer_idx, percentages in self.logit_metrics["experts_dropped"].items():
            data.extend([(layer_idx, p) for p in percentages])
        
        df = pd.DataFrame(data, columns=['Layer', 'Experts Dropped'])
        
        # Create the plot
        # plt.figure(figsize=(12, 6))
        # sns.violinplot(x='Layer', y='Experts Dropped', data=df)
        
        # plt.title('Experts dropped per Layer')
        # plt.xlabel('Layer Index')
        # plt.ylabel('Experts Dropped')
        
        # # Improve x-axis labels if there are many layers
        # if len(self.logit_metrics["experts_dropped"]) > 10:
        #     plt.xticks(rotation=45)
        
        # plt.tight_layout()
        # wandb.log({"Experts_dropped per Layer": wandb.Image(plt)})
        # plt.close()

    def plot_unique_experts_per_layer(self):
        # print("Plotting confidence percentage per layer")
        if "unique_experts" not in self.logit_metrics:
            raise KeyError("No unique experts in this layer.")
        
        # Prepare data for plotting
        data = []
        for layer_idx, percentages in self.logit_metrics["unique_experts"].items():
            data.extend([(layer_idx, p) for p in percentages])
        
        df = pd.DataFrame(data, columns=['Layer', 'Unique Experts'])
        
        # Create the plot
        # plt.figure(figsize=(12, 6))
        # sns.violinplot(x='Layer', y='Unique Experts', data=df)
        
        # plt.title('Unique Experts per Layer')
        # plt.xlabel('Layer Index')
        # plt.ylabel('Unique Experts')
        
        # # Improve x-axis labels if there are many layers
        # if len(self.logit_metrics["unique_experts"]) > 10:
        #     plt.xticks(rotation=45)
        
        # plt.tight_layout()
        # wandb.log({"Unique experts per Layer": wandb.Image(plt)})
        # plt.close()

    # @if_enabled
    # def log_confidence_scores(self, confidence_scores, layer_idx):
    #     if "confidence_scores" not in self.logit_metrics:
    #         self.logit_metrics["confidence_scores"] = {}
    #     if layer_idx not in self.logit_metrics["confidence_scores"]:
    #         self.logit_metrics["confidence_scores"][layer_idx] = []
    #     self.logit_metrics["confidence_scores"][layer_idx].append(torch.var(confidence_scores))

    # def log_confidence_scores(self, confidence_score, layer_idx):
    #     if "confidence_scores" not in self.logit_metrics:
    #         self.logit_metrics["confidence_scores"] = {}
    #     if layer_idx not in self.logit_metrics["confidence_scores"]:
    #         self.logit_metrics["confidence_scores"][layer_idx] = []
        
    #     self.logit_metrics["confidence_scores"][layer_idx].append(confidence_score)
        
    #     # Print the current state for debugging
    #     # print(f"Layer {layer_idx}: Added score {confidence_score}")
    #     # print(f"Current scores for layer {layer_idx}: {self.logit_metrics['confidence_scores'][layer_idx]}")




        
    # # @if_enabled       
    # def plot_confidence_scores(self):
    #     if "confidence_scores" not in self.logit_metrics:
    #         raise KeyError("No confidence scores logged.")
            
    #     confidence_scores = self.logit_metrics["confidence_scores"]
    #     rows, cols = 4, 8  # Adjust as necessary
    #     plt.figure(figsize=(20, 12))
    #     for layer_idx, scores in confidence_scores.items():
    #         plt.subplot(rows, cols, layer_idx + 1)
    #         scores_np = [score.item() for score in scores]
    #         plt.plot(scores_np)
    #         plt.title(f"Layer {layer_idx}")
    #         plt.xlabel("Batch Index")
    #         plt.ylabel("Variance")
    #         plt.tight_layout()

    #     wandb.log({"Confidence Scores for All Layers": wandb.Image(plt)})
    #     plt.close()

    def log_confidence_scores(self, confidence_score, layer_idx):
        if "confidence_scores" not in self.logit_metrics:
            self.logit_metrics["confidence_scores"] = {}
        if layer_idx not in self.logit_metrics["confidence_scores"]:
            self.logit_metrics["confidence_scores"][layer_idx] = []
        
        # Convert CUDA tensor to CPU numpy array
        if isinstance(confidence_score, torch.Tensor):
            if confidence_score.is_cuda:
                confidence_score = confidence_score.cpu()
            confidence_score = confidence_score.detach().numpy()
        
        self.logit_metrics["confidence_scores"][layer_idx].append(confidence_score)

    def plot_confidence_scores(self):
        if not self.logit_metrics.get("confidence_scores"):
            print("No confidence scores to plot.")
            return

        # plt.figure(figsize=(20, 12))
        # for layer_idx, scores in self.logit_metrics["confidence_scores"].items():
            # plt.plot(scores, label=f"Layer {layer_idx}")
        
        # plt.title("Confidence Scores Across Layers")
        # plt.xlabel("Step")
        # plt.ylabel("Confidence Score")
        # plt.legend()
        # plt.tight_layout()

        # wandb.log({"Confidence Scores for All Layers": wandb.Image(plt)})
        # plt.close()
            
        

    ###functions to dump the router logits and modified logits        
    @if_enabled
    def dump_router_logits(self, router_logits, layer_idx):
        if router_logits.is_cuda:
            router_logits = router_logits.cpu()
        self.router_logit_store.append((self.batch_idx, layer_idx, router_logits))
    
    @if_enabled
    def dump_modified_logits(self, router_logits, layer_idx):
        if router_logits.is_cuda:
            router_logits = router_logits.cpu()
        self.modified_logit_store.append((self.batch_idx, layer_idx, router_logits))
        
    @if_enabled
    def log_router_logits(self, orig_router_logits, mod_router_logits, layer_idx):
        # print("The values of batch index: ", self.batch_idx, "profile complete flag: ", self.profile_complete, "logit_logging_frequency: ", self.logit_logging_frequency)
        if self.profile_complete:
            if self.record_next_batch:
                if not self.is_prefill:
                    self.dump_router_logits(orig_router_logits, layer_idx)
                    self.dump_modified_logits(mod_router_logits, layer_idx)
                    self.record_next_batch = False
                else: 
                    self.record_next_batch = True
                    # print("We encountered a prefill batch, let's try the next one")
                    
            if self.batch_idx % self.logit_logging_frequency == 0:
                if not self.is_prefill:
                    self.dump_router_logits(orig_router_logits, layer_idx)
                    self.dump_modified_logits(mod_router_logits, layer_idx)
                else:
                    self.record_next_batch = True
    ##############################################################
    ## Functions that log the number of experts dropped and the experts that were dropped            
    @if_enabled
    def log_dropped_experts(self, reassigned_experts, experts_dropped_count, layer_idx):
        # print("The values of batch index: ", self.batch_idx, "profile complete flag: ", self.profile_complete, "logit_logging_frequency: ", self.logit_logging_frequency)
        if self.profile_complete:
            if self.record_next_batch:
                if not self.is_prefill:
                    self.dump_experts_dropped(experts_dropped_count, layer_idx, self.batch_idx)
                    self.dump_which_experts_dropped(reassigned_experts, layer_idx, self.batch_idx)
                    self.record_next_batch = False
                else: 
                    self.record_next_batch = True
                    # print("We encountered a prefill batch, let's try the next one")
                    
            if self.batch_idx % self.logit_logging_frequency == 0:
                if not self.is_prefill:
                    self.dump_experts_dropped(experts_dropped_count, layer_idx, self.batch_idx)
                    self.dump_which_experts_dropped(reassigned_experts, layer_idx, self.batch_idx)
                    self.record_next_batch = False
                else:
                    self.record_next_batch = True
                
    @if_enabled
    def dump_experts_dropped(self, experts_dropped_in_layer_in_batch, layer_idx, batch_idx):
        if "dropped_logits" not in self.logit_metrics:
            self.logit_metrics["dropped_logits"] = {}
        if layer_idx not in self.logit_metrics["dropped_logits"]:
            self.logit_metrics["dropped_logits"][layer_idx] = CDFSketch(metric_name=f"dropped_logits_layer_{layer_idx}_batch_{batch_idx}", save_table_to_wandb=False)
        self.logit_metrics["dropped_logits"][layer_idx].put(experts_dropped_in_layer_in_batch)
        #TODO: Add function that also captures the number of experts dropped for the whole batch and also between iterations
    
    @if_enabled
    def dump_which_experts_dropped(self, experts_reassigned, layer_idx, batch_idx):
        if "dropped_experts" not in self.logit_metrics:
            self.logit_metrics["dropped_experts"] = {}
        if layer_idx not in self.logit_metrics["dropped_experts"]:
            self.logit_metrics["dropped_experts"][layer_idx] = {}

        for exp_id in experts_reassigned:
            if exp_id not in self.logit_metrics["dropped_experts"][layer_idx]:
                self.logit_metrics["dropped_experts"][layer_idx][exp_id] = 1
            self.logit_metrics["dropped_experts"][layer_idx][exp_id] += 1
        
    @if_enabled
    def clear_router_logits(self):
        self.router_logit_store = []
        self.modified_logit_store = []
        self.expert_counts = []
    
    @if_enabled  
    def plot_reassigned_experts(self):
        dropped_experts = self.logit_metrics["dropped_experts"]
        for layer_idx, experts in dropped_experts.items():
            expert_ids = list(experts.keys())
            reassignment_counts = list(experts.values())
            
            # Ensure reassignment_counts are on CPU and converted to Python numbers if they are tensors
            reassignment_counts = [count.cpu().item() if isinstance(count, torch.Tensor) else count for count in reassignment_counts]
            expert_ids = [exp_id.cpu().item() if isinstance(exp_id, torch.Tensor) else exp_id for exp_id in expert_ids]

            # Create a DataFrame from the lists
            df = pd.DataFrame({'Expert IDs': expert_ids, 'Reassignment Count': reassignment_counts})

            # Create a bar chart using Plotly Express
            # fig = px.bar(
            #     df,
            #     x='Expert IDs',
            #     y='Reassignment Count',
            #     labels={'x': 'Expert IDs', 'y': 'Reassignment Count'},
            #     title=f'Reassignment Counts for Layer {layer_idx}'
            # )
            # fig.update_layout(xaxis_title='Expert IDs', yaxis_title='Reassignment Count')
            
            # # Save the figure to a file
            # fig.write_image(f"{self.full_path}/reassignment_counts_layer_{layer_idx}.png")

            # # Log the plot to wandb
            # wandb.log({f"Layer {layer_idx} Reassignment Counts": fig})
            
    @if_enabled
    def log_impact_of_new_policy(self, probs_orig, probs_new, layer_idx):
        if "impact_of_new_policy" not in self.logit_metrics:
            self.logit_metrics["impact_of_new_policy"] = {}
        if layer_idx not in self.logit_metrics["impact_of_new_policy"]:
            self.logit_metrics["impact_of_new_policy"][layer_idx] = []
        self.logit_metrics["impact_of_new_policy"][layer_idx].append(torch.mean(probs_new - probs_orig))
        
    @if_enabled
    def plot_impact_of_new_policy(self):
        if "impact_of_new_policy" not in self.logit_metrics:
            raise KeyError("No impact of new policy logged.")
            
        impact_of_new_policy = self.logit_metrics["impact_of_new_policy"]
        # Loop through each layer and plot the data on the same figure
        rows, cols = 4, 8  # Adjust as necessary
        # plt.figure(figsize=(20, 12))
        # for layer_idx, impacts in impact_of_new_policy.items():
        #     plt.subplot(rows, cols, layer_idx + 1)
        #     impacts_np = [impact.item() for impact in impacts]
        #     plt.plot(impacts_np)
        #     plt.title(f"Layer {layer_idx}")
        #     plt.xlabel("Batch Index")
        #     plt.ylabel("Mean Impact")
        #     plt.tight_layout()

        # wandb.log({"Impact of New Policy for All Layers": wandb.Image(plt)})
        # plt.close()
            
    @if_enabled
    def log_grouping_info(self, percentage_in_high_confidence, layer_idx):
        if "group_labels" not in self.logit_metrics:
            self.logit_metrics["group_labels"] = {}
        if layer_idx not in self.logit_metrics["group_labels"]:
            self.logit_metrics["group_labels"][layer_idx] = []
        self.logit_metrics["group_labels"][layer_idx].append(percentage_in_high_confidence)

        
    @if_enabled
    def plot_grouping_info(self):
        if "group_labels" not in self.logit_metrics:
            raise KeyError("No grouping info logged.")
            
        group_labels = self.logit_metrics["group_labels"]
        
        # Loop through each layer and plot the single value as a scatter point
        # Create a new figure
        # Define the number of rows and columns for subplots
        rows, cols = 4, 8  # Adjust as necessary
        # Grouping Info for All Layers
        # plt.figure(figsize=(20, 12))
        # for layer_idx, labels in group_labels.items():
        #     plt.subplot(rows, cols, layer_idx + 1)
        #     labels_np = [label.item() for label in labels]
        #     plt.plot(labels_np)
        #     plt.title(f"Layer {layer_idx}")
        #     plt.xlabel("Batch Index")
        #     plt.ylabel("High Conf %")
        #     plt.tight_layout()

        # wandb.log({"Grouping Info for All Layers": wandb.Image(plt)})
        # plt.close()
        
    # @if_enabled
    def plot_metrics(self):
        # print all the keys in the logit_metrics dictionary
        # self.write_to_csv()
        # self.clear_router_logits()
        # self.plot_reassigned_experts()
        print("the plotting function is called but nothing is being plotted")
        # self.plot_conf_percentage_per_layer()
        # self.plot_experts_dropped_per_layer()
        # self.plot_unique_experts_per_layer()
        # self.plot_grouping_info()
        # self.plot_impact_of_new_policy()
        # print(self.logit_metrics.keys())
        # print(self.logit_metrics)
        # wandb.init(project="MLSys_Prowl", group="Dropping policies", name=f'{self.num_experts}_experts')
        # for layer_idx in range(len(self.logit_metrics["dropped_logits"])):
        #         self.logit_metrics["dropped_logits"][layer_idx].plot_cdf(self.full_path, f"dropped_logits_layer_{layer_idx}", "experts_dropped")
            
            
            # if task_type == "dropped_logits":
            #     for layer_idx, cdf_sketch in measurements.items():
            #         cdf_sketch.plot_cdf(self.full_path, cdf_sketch._metric_name)
        # wandb.finish()
        # Write router logits to CSV
        
        
    @if_enabled    
    def write_to_csv(self):
        original_logits_df = self._create_logit_dataframe(self.router_logit_store)
        modified_logits_df = self._create_logit_dataframe(self.modified_logit_store)
        original_logits_df.to_csv(os.path.join(self.full_path, f"router_logits.csv"), index=False)
        modified_logits_df.to_csv(os.path.join(self.full_path, f"router_logits_modified.csv"), index=False)
        self.save_logit_visualization_to_wandb(original_logits_df, modified_logits_df)
        
    @if_enabled    
    def _create_logit_dataframe(self, logit_store):
        """Helper method to create a DataFrame from the logit store."""
        # Assuming a maximum number of logits, you can adjust this as needed
        max_logits = 8  # Example, change this based on your maximum number of logits
        
        # Initialize a list to collect all rows
        data = []
        
        # Process each logit entry in the store
        for batch_idx, layer_idx, logits_tensor in logit_store:
            logits_list = logits_tensor.tolist()  # Convert tensor to a list
            for logit in logits_list:
                # Append the combined data to the list
                data.append([batch_idx, layer_idx] + logit)
        
        # Create a DataFrame from the collected data
        columns = ['Batch Index', 'Layer Index'] + [f'Logit_{i}' for i in range(max_logits)]
        df = pd.DataFrame(data, columns=columns)
        
        return df

    @if_enabled
    def visualize_and_save_layer_data(self, layer_index, batch_index, original_df, modified_df):
        original_layer_df = original_df[(original_df['Layer Index'] == layer_index) & (original_df['Batch Index'] == batch_index)]
        modified_layer_df = modified_df[(modified_df['Layer Index'] == layer_index) & (modified_df['Batch Index'] == batch_index)]
        
        # if batch_index == 15 and layer_index == 31:
        #     print(original_layer_df)
        #     print(modified_layer_df)

        # Adjust the columns to include all logits
        original_logits = original_layer_df.iloc[:, 2:]
        modified_logits = modified_layer_df.iloc[:, 2:]

        global_min = min(original_logits.replace(-np.inf, np.nan).min().min(), modified_logits.replace(-np.inf, np.nan).min().min())
        global_max = max(original_logits.replace(-np.inf, np.nan).max().max(), modified_logits.replace(-np.inf, np.nan).max().max())

        if global_min == global_max:
            global_max += 1e-9  # Adding a small epsilon to avoid singular transformation

        # plt.figure(figsize=(24, 10))
        # ax1 = plt.subplot(1, 2, 1)
        # sns.heatmap(original_logits.replace(-np.inf, np.nan), annot=False, fmt=".2f", cmap="vlag", ax=ax1, vmin=global_min, vmax=global_max)
        # ax1.set_xticklabels(original_logits.columns, rotation=45, ha='right')
        # ax1.set_title(f'Original Logits for Layer {layer_index}')

        # ax2 = plt.subplot(1, 2, 2)
        # sns.heatmap(modified_logits.replace(-np.inf, np.nan), annot=False, fmt=".2f", cmap="vlag", ax=ax2, vmin=global_min, vmax=global_max)
        # ax2.set_xticklabels(modified_logits.columns, rotation=45, ha='right')
        # ax2.set_title(f'Modified Logits for Layer {layer_index}')

        # plt.tight_layout()
        # wandb.log({f"layer_{layer_index}_batch_{batch_index}_logits_comparison": wandb.Image(plt)})
        # plt.close()
        
    @if_enabled    
    def save_logit_visualization_to_wandb(self, original_logits_df, modified_logits_df):
        # get all unique values of batch_idx from the df and iterate over them
        batch_indices = original_logits_df['Batch Index'].unique()
        for batch_idx in batch_indices:
            for layer_index in range(32):
                self.visualize_and_save_layer_data(layer_index, batch_idx, original_logits_df, modified_logits_df)
        