# Survival analysis as a classification problem - Unit Test
# Author: Alex (Oleksiy) Varfolomiyev

import unittest
from stacking import load_data
from stacking import ClassificationModel

class ClassificationModelTest(unittest.TestCase):
    def setUp(self):
        # Create a toy data set on which you may test your implementation.
        from sklearn.model_selection import train_test_split
        n = 200  # Number of observations
        p = 5  # Number of features
        self.X, self.y = load_data(n, p)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                test_size=0.2, random_state=42)


    """def test_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        model = ClassificationModel(model_name='logistic_regression')
        model.train_test(self.X_train, self.y_train)
        accuracy = model.evaluate_test(self.X_test, self.y_test)
        self.assertAlmostEqual(accuracy, .99, delta = 0.01)"""

    def test_random_forest(self):
        model = ClassificationModel(model_name='random_forest')
        model.train_test(self.X_train, self.y_train)
        accuracy = model.evaluate_test(self.X_test, self.y_test)
        self.assertAlmostEqual(accuracy, .98, delta = 0.01)

#######################################################################
# run unit test for the classification class
#######################################################################

if __name__ == '__main__':
    unittest.main()


