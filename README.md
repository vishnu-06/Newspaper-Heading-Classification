# Text-classification-of-Newspaper-Heading

Despite its extensive content, unstructured text can be difficult to analyze and extract insights from due to its lack of structure. Almost any type of text can be organized, structured, and categorised with text classifiers. Text classification is widely used to analyse this unstructured text and classify them based on common topics. There are many sites that produce a lot of news every day today on the Internet. Additionally, user demand for information has been steadily increasing, thus it is critical that the news be classified to enable users to quickly and effectively obtain the information of interest. On this basis, the user's former interests could be utilised to discover subjects of untracked news and/or to generate customised suggestions using the machine learning model for automatic news classification. Hence, we propose creating an application for text classification that accurately classifies newspaper titles by topic. We aim to create a text classification application using word embedding and neural network algorithms.

## Flow Diagram
![image](https://user-images.githubusercontent.com/70327869/201008053-b9c538e6-0e74-4eac-944a-310c8c4654a4.png)

## GRU Model
![image](https://user-images.githubusercontent.com/70327869/201008232-5ca21116-c744-4f47-9597-ee6aa29a62c7.png)

## Evaluation Metrics
![image](https://user-images.githubusercontent.com/70327869/201008332-5c8a29ad-275f-4f06-9a4a-0a77d42d580c.png)

## Confusion Matrix
![image](https://user-images.githubusercontent.com/70327869/201008393-e5a24cc2-9211-4062-98ad-12d69791c1ca.png)

## Reciever Operating Characteristic Curve
![image](https://user-images.githubusercontent.com/70327869/201008483-fe591050-fc8d-4f1b-8751-f99c729acbee.png)

## Training Time
![image](https://user-images.githubusercontent.com/70327869/201008575-b9f00ec9-e531-487f-804b-a96d088dcecb.png)

## Accuracy Comparison
![image](https://user-images.githubusercontent.com/70327869/201008655-75b518b9-c228-4c34-a051-067217ca9407.png)

## Results

We achieve a text-classification accuracy of 84% with 50 epochs using Bi-directional GRU with attention mechanism on 10 newspaper categories. We have achieved accuracies comparable to that of LSTM with 14% lesser training time. For future work, we aim to expand our categories and further increase the accuracy by increasing the number of epochs. 
