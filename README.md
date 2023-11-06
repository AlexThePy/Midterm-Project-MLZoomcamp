# Midterm-Project-MLZoomcamp
In this project, I chose a dataset about mobile phone features so that I can predict their prices

Link to project: https://www.kaggle.com/datasets/rkiattisak/mobile-phone-price



Mobile Phone Price Prediction Project
Description
This project aims to predict the prices of mobile phones based on a comprehensive dataset that includes various specifications such as Brand, Model, Storage, RAM, Screen Size (inches)	Camera (MP),	Battery Capacity (mAh), Price ($).
This tool is designed to aid  consumers, retailers, and manufacturers in understanding the price dynamics in the mobile phone market and making informed decisions.

I have initially cleaned my csv file so that the Camera (MP) column will only show the main camera.

Metadata
Source Information
Source of Data:
The dataset utilized in this project was synthesized by an advanced Large Language Model (LLM). It was programmatically generated to reflect plausible mobile phone specifications and their corresponding market prices. It is important to note that the data does not originate from real-world market research or direct manufacturer information.

1.Purpose of Synthetic Data:
The use of an LLM-generated dataset served multiple purposes:

2.Feasibility and Prototyping: It enabled the development and testing of the machine learning pipeline without the constraints of data acquisition costs and privacy concerns associated with real-world data.

3.Educational Value: The project illustrates the end-to-end process of a machine learning task, including data handling, model training, and deployment, which remains valid and educational regardless of the data origin.

4.Research and Experimentation: The synthetic dataset provided a controlled environment to explore the potential correlations and patterns that a machine learning model might learn and how it generalizes to data that mimics real-world distributions.

5.Credibility and Limitations:While the LLM strives to produce realistic data based on patterns learned from vast text corpora, the synthetic nature of the dataset inherently comes with limitations:

6.Authenticity: The generated data may not accurately represent current market trends or specific brand pricing strategies.

7.Variability and Bias: The diversity of the dataset might be limited to the LLM's training data, and unexpected biases could be introduced.

8.Validation and Testing: The absence of true market validation means that the model's predictive performance, as evaluated on this dataset, does not guarantee similar results on real-world data


9.Intended Use:
The model and its findings should be considered a demonstration of machine learning techniques suitable for educational purposes and not as a basis for real-world business decisions or academic conclusions about the mobile phone market.

10.Further Research:
For any production or research-oriented application, it is recommended to utilize verified datasets obtained from actual mobile phone sales data and market analysis to ensure the accuracy and relevance of the model's insights.



Description of Features

• Brand: the manufacturer of the phone
• Model: the name of the phone model
• Storage (GB): the amount of storage space (in gigabytes) available on the phone
• RAM (GB): the amount of RAM (in gigabytes) available on the phone
• Screen Size (inches): the size of the phone's display screen in inches
• Camera (MP): the megapixel count of the phone's main rear camera
• Battery Capacity (mAh): the capacity of the phone's battery in milliampere hours
• Price ($): the retail price of the phone in US dollars
Each row represents a different mobile phone model. The data can be used to analyze pricing trends and compare the features and prices of different mobile phones.


INSTRUCTIONS ON HOW TO RUN THE CODE:

-Download mobile_price.csv from this project
-Install Jupyter Notebook (if you haven't already, you can install Anaconda which will install Jupyter Notebook as well)
-Use the code from notebook.jpynb or script.py to run the code in a Jupyter Notebook (after opening Jupyter, File>New Notebook>Pyhton 3 (jpykernel)) and run the code
-In the first cells of code you will see how the dataset is structured and the distribution of prices
-Observe how I've used the Storage, Screen Size and Camera (MP) features to generate a histogram with Predictions vs actual distribution
-The model was deployed with Flask using Jupyter Notebook, thus the reason for importing the ''Thread'' library. 
