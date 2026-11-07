# DSC180 Capstone

Pain Axis Reproduction and Activation Steering

UC San Diego | DSC 180A/B Capstone | 2026–2027

Team Members: Terence Zhang, Alexander Kuo

Advisors: Misha Belkin, Tianhao Wang


Project Overview:

This project reproduces and extends methods from The Pain Axis: LLMs Represent Self-Directed Harm and Act on It. We investigate whether large language models encode pain-related representations in their internal activations and whether activation steering can influence model behavior.

Our work focuses on extracting pain-related steering directions using Difference-in-Means (DiM) and investigating Recursive Feature Machines (RFM) as an alternative method.

Original implementation: https://github.com/valen-research/Pain-axis


Methodology:

Our initial experiments use Qwen2.5-1.5B-Instruct on UCSD DSMLP.

We extract final-token and mean-token hidden activations across all 28 transformer layers using first-person and third-person prompts from the original paper's datasets.

Pain directions are constructed using PCA-denoised Difference-in-Means and evaluated through five-fold cross-validation grouped by sentence set.

We also implement activation steering by modifying the model's hidden representations during generation.


AI use statement:

Claude was used to assist with code development, debugging, troubleshooting technical issues, and documentation. All experiments were conducted and results evaluated by the project team.


References

The Pain Axis: LLMs Represent Self-Directed Harm and Act on It — Original GitHub repository

Beaglehole et al., Toward Universal Steering and Monitoring of AI Models.