# Import KNN model
from sklearn.neighbors import KNeighborsClassifier

# Add KNN to the list of classifiers
classifiers['K-Nearest Neighbors'] = KNeighborsClassifier()

# Iterate through classifiers to train and test the models
for name, clf in classifiers.items():
    # Build a corpus from the 'Test Case/v2' column
    corpus = df['v2'].tolist()

    # Create a Bag-of-Words model using CountVectorizer and TfidfTransformer
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    
    # Create a pipeline with a text feature vectorizer (CountVectorizer), TfidfTransformer, and the classifier
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('transformer', transformer),
        ('classifier', clf)
    ])
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(corpus, df['v1'], test_size=0.3, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    print(f'\nResults for {name}:')
    print(f'Accuracy: {accuracy}')
    print('Classification Report:\n', report)
