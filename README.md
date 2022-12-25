# Thermochemical Stability Prediction Project

Gene Sequence Stability Analysis and Prediction Project

How to Run:

1. Clone the repo
2. Open a new terminal and the run the command below:
'''python
streamlit run main.py
'''

Thermostability refers to the ability of an enzyme to maintain its activity and structural integrity under high temperatures. Enzymes are proteins that catalyze chemical reactions in living cells, and they are typically optimally active within a certain temperature range. However, some enzymes are able to retain their activity and structure at higher temperatures, making them thermostable.

There are several factors that contribute to the thermostability of enzymes. One factor is the stability of the enzyme's active site, which is the region where substrate molecules bind and the chemical reaction takes place. The active site may be stabilized by the presence of specific amino acids or by the overall structure of the enzyme.

Another factor is the stability of the enzyme's protein structure as a whole. Enzymes are made up of long chains of amino acids that are folded into a specific three-dimensional structure. This structure can be disrupted by high temperatures, leading to denaturation of the protein and loss of activity. However, some enzymes have a more stable structure that is less prone to denaturation at high temperatures.

Thermostable enzymes are found in a variety of organisms, including bacteria that live in hot springs and other extreme environments. These enzymes have evolved to function efficiently at high temperatures and are used in a variety of industrial and research applications.

Machine learning is a type of artificial intelligence that involves the use of algorithms and statistical models to enable computers to learn and make predictions or decisions based on data. In the context of enzyme stability prediction, machine learning algorithms can be used to analyze large datasets of enzyme properties and performance data to identify patterns and trends that may not be immediately apparent to a human analyst.

For example, a machine learning model could be trained on a dataset of enzyme sequences and stability data, and then be used to predict the thermostability of a new enzyme based on its sequence alone. The model could use various techniques such as regression or classification to make these predictions.

There are several advantages to using machine learning for enzyme stability prediction. One advantage is that it can be faster and more accurate than manual analysis, especially for large datasets. Additionally, machine learning models can learn from and adapt to new data, which means that they can become more accurate over time as more data is collected. This can be particularly useful for predicting the stability of enzymes that are less well-studied or for which there is limited data available.

Overall, machine learning has the potential to significantly improve our understanding of enzyme stability and to facilitate the design and optimization of enzymes for various applications.

In this project, we will use PyTorch and XGBoost Regressors to regress the thermostability values of enzymes given the pH and Amino Acid strands.
