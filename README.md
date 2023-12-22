 ## EcoBin: Waste Classification Model

### Overview
EcoBin is a waste classification model that utilizes transfer learning and data augmentation techniques to accurately categorize waste items into six different classes: biodegradable, cardboard, glass, metal, paper, and plastic. The model is trained on a large dataset of waste images and can be used to help individuals and organizations properly sort their waste for recycling and disposal.

### Key Features
- Utilizes transfer learning from a pre-trained EfficientNetV2S model, which provides a strong foundation for image classification tasks.
- Employs data augmentation techniques such as random flipping, rotation, zooming, contrast adjustment, and translation to enhance the model's robustness and generalization capabilities.
- Achieves high accuracy in classifying waste items, making it a valuable tool for waste management and recycling efforts.

### Usage
To use the EcoBin model, follow these steps:

1. **Install the required dependencies:**
   - TensorFlow: `2.13.0`
   - Keras: `2.13.1`
   - NumPy: `1.24.3`
   - Matplotlib: `1.24.3`
   - Scikit-learn: `1.3.2`

2. **Load the pre-trained model:**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('ecobin_model.h5')
   ```

3. **Preprocess the input image:**
   - Resize the image to the model's input size.
   - Apply any necessary data augmentation techniques.

4. **Make a prediction:**
   ```python
   prediction = model.predict(preprocessed_image)
   ```

5. **Interpret the prediction:**
   The model will output a probability distribution over the six waste classes. The class with the highest probability is the predicted class for the input image.

### Dataset Information:
#### Class Distribution:

| Class          | Total Photos |
| -------------- | ------------ |
| Biodegradable | 2221         |
| Cardboard      | 1867         |
| Glass          | 3176         |
| Metal          | 1660         |
| Paper          | 2422         |
| Plastic        | 1645         |

- **Total Classes:** 6
- **Total Photos:** 12,991

#### Dataset Split:

- Training Set: 70%
- Validation Set: 20%
- Test Set: 10%

#### Class Distribution in Each Set:

| Class         | Train Set | Validation Set | Test Set | Total |
|---------------|-----------|-----------------|----------|-------|
| Biodegradable | 1,554     | 444             | 223      | 2,221 |
| Cardboard     | 1,306     | 374             | 187      | 1,867 |
| Glass         | 2,223     | 635             | 318      | 3,176 |
| Metal         | 1,161     | 332             | 167      | 1,660 |
| Paper         | 1,696     | 483             | 243      | 2,422 |
| Plastic       | 1,151     | 329             | 165      | 1,645 |

### Evaluation
The EcoBin model was evaluated on a held-out test set and achieved an accuracy of 94%.
- **Validation Loss:** 0.1578
- **Validation Accuracy:** 94.40%
#### This indicates the model's ability to accurately classify waste items on unseen data, showcasing its robustness and effectiveness.
![Epoch Accuracy and Loss](https://github.com/Ecobin-Capstone/ecobin-waste-classifier/assets/103976140/cc4085c8-305b-4eef-9119-8e76ea4a5c9a)

### Applications
The EcoBin model can be used in a variety of applications, including:

- **Waste sorting:** The model can be integrated into waste sorting systems to help individuals and organizations properly sort their waste for recycling and disposal.
- **Waste management:** The model can be used to analyze waste streams and identify trends in waste generation and composition. This information can be used to improve waste management practices and reduce the amount of waste sent to landfills.
- **Education and awareness:** The model can be used to educate the public about different types of waste and the importance of proper waste management.

### Conclusion
The EcoBin model is a powerful tool for waste classification that can be used to improve waste management practices and reduce the environmental impact of waste. The model's high accuracy and ease of use make it a valuable asset for individuals, organizations, and communities looking to make a positive impact on the environment.
